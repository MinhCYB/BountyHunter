"""
================================================================================
MODULE: DATA PREPROCESSING PIPELINE (TIỀN XỬ LÝ DỮ LIỆU)
================================================================================

Mục đích:
Chuẩn hóa, làm sạch và tổng hợp dữ liệu thô thành:
  (A) Master Table theo từng ngày — phục vụ Feature Engineering và huấn luyện mô hình ML.
  (B) Data Marts chuyên đề (Customer, Product, Promo) — phục vụ phân tích EDA & BI.

Cấu trúc 4 Lớp (4-Layer Architecture):
  Lớp 1 — Data Ingestion & Casting:
      Đọc CSV thô, ép kiểu DateTime và ID string theo config, lưu Parquet.

  Lớp 2 — Promotional Feature Engineering:
      Tái tính chiết khấu kỳ vọng (expected_discount) theo đúng luật nghiệp vụ,
      đo lường Revenue Leakage với tolerance threshold để loại float precision noise.

  Lớp 3 — Cluster Aggregation:
      Gom nhóm dữ liệu thành 3 cụm độc lập theo trục ngày (Date):
        - Transaction Cluster : Doanh thu, chiết khấu, leakage.
        - Customer Cluster    : Hoàn trả, đánh giá sản phẩm.
        - Operation Cluster  : Lưu lượng web, tồn kho.

  Lớp 4 — Assembly & Distribution:
      4A. build_master_table()   — Left-join tất cả cluster vào Date Spine → Master Table.
      4B. build_all_data_marts() — Xuất 3 Data Mart chuyên đề từ engineered items.

Dependency quan trọng:
  build_all_data_marts() phụ thuộc vào file _temp_engineered_items.parquet được tạo
  bởi _process_transaction_cluster() (chạy bên trong build_master_table()).
  Luôn gọi build_master_table() trước build_all_data_marts(), hoặc dùng run_prep()
  để đảm bảo thứ tự thực thi đúng.

Đầu ra (Output):
  config.PROCESSED/master_table.parquet             — Master Table cho ML
  config.PROCESSED/mart_customer_cohort.parquet     — Data Mart: Cohort & CLV
  config.PROCESSED/mart_product_performance.parquet — Data Mart: Hiệu suất sản phẩm
  config.PROCESSED/mart_promo_campaign.parquet      — Data Mart: Hiệu quả khuyến mãi

  [NOTE] web_traffic bắt đầu 2013-01-01, sales từ 2012-07-04 → gap ~6 tháng đầu
         traffic = NaN, được điền 0 qua zero_fill_cols (không có session = 0).
  [NOTE] promo_id_2 null ~100% nhưng profiling cho thấy có 2 unique values
         (PROMO-0015, PROMO-0025) → logic hiện tại xử lý đúng sau khi fix ID casting.
  [NOTE] min_order_value = 0 trong promotions là hợp lệ ("không có ngưỡng tối thiểu"),
         p1_mov_mask đã được cập nhật để chỉ flag khi min_order_value > 0.
"""

import pandas as pd
import numpy as np
from src.config import config

logger = config.get_logger(__name__)

# Các hằng số schema và ngưỡng sai số được quản lý tập trung tại src/config.py:
#   config.MATH_TOLERANCE_MIN  — ngưỡng float noise VND  (≡ config.MATH_TOLERANCE_MIN cũ)
#   config.STRING_ID_COLS      — cột ID string cần chuẩn hóa
#   config.CONSTANT_COLS       — cột hằng số cần drop


# ==========================================
# LỚP 1: TẢI VÀ ÉP KIỂU DỮ LIỆU
# ==========================================

def _cast_data_types(name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Chuẩn hóa kiểu dữ liệu cho một DataFrame dựa trên cấu hình dự án.

    Thực hiện 4 bước theo thứ tự:

    1. Ép kiểu DateTime: Với mỗi cột ngày tháng được khai báo trong config.DATE_COLS,
       chuyển đổi sang pd.Timestamp bằng errors='coerce' — giá trị không parse được
       sẽ thành NaT thay vì raise exception.

    2. Chuẩn hóa ID string [FIX v3 — Silent Merge Failure]:
       Pandas tự động ép cột object có chứa NaN sang float64 khi đọc CSV
       (VD: promo_id "PROMO-0014" mixed với NaN → đọc thành "PROMO-0014" OK,
       nhưng nếu cột là số như order_id thì "123" → 123.0 dưới dạng str là "123.0").
       Khi merge, key "123" ≠ "123.0" → join miss → rớt sạch dữ liệu.
       Fix: Quét tất cả cột thuộc config.STRING_ID_COLS, ép về str, cắt đuôi '.0'
       bằng regex, đặt lại NaN cho các giá trị "nan" và "".

    3. Chuẩn hóa cột text đặc biệt:
       - applicable_category: NaN mang nghĩa "áp dụng tất cả danh mục" → fillna("All").

    4. Drop cột CONSTANT [ADD v3]:
       reorder_flag toàn bộ = 0 (profiling: Unique=1, Range: 0→0).
       Không mang thông tin, drop để tránh noise feature matrix.

    Args:
        name (str): Tên bảng (CSV stem), dùng để tra cứu config.DATE_COLS.
        df (pd.DataFrame): DataFrame cần chuẩn hóa.

    Returns:
        pd.DataFrame: DataFrame đã được chuẩn hóa kiểu dữ liệu.
    """
    # Bước 1: Ép kiểu DateTime
    if name in config.DATE_COLS:
        for col in config.DATE_COLS[name]:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    # Bước 2: Chuẩn hóa ID string — tránh Silent Merge Failure
    for col in config.STRING_ID_COLS:
        if col in df.columns:
            df[col] = (
                df[col].astype(str)
                    .str.replace(r"\.0$", "", regex=True)
                    .str.strip()
                    .replace("nan", np.nan)
                    .replace("",    np.nan)
            )

    # Bước 3: Chuẩn hóa text đặc biệt
    if "applicable_category" in df.columns:
        df["applicable_category"] = df["applicable_category"].fillna("All").astype(str)

    # Bước 4: Drop cột CONSTANT
    cols_to_drop = [c for c in config.CONSTANT_COLS if c in df.columns]
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)
        logger.info(f"   [{name}] Drop cột CONSTANT: {cols_to_drop}")

    return df


def initialize_parquet_storage(source_tables: list):
    """
    Đọc toàn bộ CSV thô, ép kiểu và lưu dưới dạng Parquet để tối ưu hiệu suất I/O.

    Duyệt qua tất cả file .csv trong config.RAW theo thứ tự alphabet. Với mỗi file
    có tên (stem) xuất hiện trong source_tables, pipeline sẽ:
      1. Đọc CSV với low_memory=False để tránh DtypeWarning.
      2. Chuẩn hóa kiểu dữ liệu qua _cast_data_types().
      3. Ghi ra Parquet tại config.PROCESSED/{table_name}.parquet.

    Mỗi bảng được bọc trong try/except — lỗi ở một bảng sẽ được log và bỏ qua.

    Args:
        source_tables (list[str]): Danh sách tên bảng cần xử lý (không có đuôi .csv).

    Side Effects:
        Ghi các file .parquet vào thư mục config.PROCESSED.
    """
    logger.info("Khởi tạo không gian lưu trữ Parquet...")
    for csv_file in sorted(config.RAW.glob("*.csv")):
        table_name = csv_file.stem
        if table_name in source_tables:
            try:
                df = pd.read_csv(csv_file, low_memory=False)
                df = _cast_data_types(name=table_name, df=df)
                df.to_parquet(config.PROCESSED / f"{table_name}.parquet", index=False)
                logger.info(f" -> Bảng [{table_name}] OK. Shape: {df.shape}")
            except Exception as e:
                logger.error(f"Lỗi xử lý bảng [{table_name}]: {e}")


# ==========================================
# LỚP 2: KHAI PHÁ ĐẶC TRƯNG KHUYẾN MÃI (PROMO LEAKAGE)
# ==========================================

def _engineer_promotional_features(order_items: pd.DataFrame, promotions: pd.DataFrame) -> pd.DataFrame:
    """
    Tái tính chiết khấu kỳ vọng theo đúng luật nghiệp vụ và đo lường Revenue Leakage.

    Hàm giữ nguyên discount_amount gốc (dòng tiền thực tế), tạo thêm expected_discount
    và revenue_leakage để định lượng thất thoát do vi phạm quy tắc khuyến mãi.

    Logic xử lý 6 bước:

    Bước 1 — Tính order_val:
        Tổng giá trị đơn hàng = sum(quantity x unit_price) theo order_id.
        Lưu ý: min_order_value = 0 là hợp lệ ("không có ngưỡng"), không phải NaN.

    Bước 2 — Join thông tin Promo:
        Join promotions 2 lần (p1_ mã chính, p2_ mã phụ).
        Sau khi _cast_data_types() fix ID string, merge không còn silent failure.

    Bước 3 — Xác định vi phạm:
        - abuse_mask  : Có p2 nhưng p1 có stackable_flag = 0.
        - p1_mov_mask : p1 tồn tại, min_order_value > 0,
                        nhưng order_val < p1_min_order_value.
                        [FIX v3] Loại trừ min_order_value = 0 (không có ngưỡng).
        - p2_mov_mask : Tương tự cho p2.

    Bước 4 — Tính expected_discount:
        Công thức từ data dictionary:
          - percentage: quantity x unit_price x (discount_value / 100)
          - fixed      : quantity x discount_value
        Vi phạm MOV hoặc Stackable → expected discount của mã đó = 0.
        Tách riêng promo_discount_p1 / p2 để mart tính ROI chính xác.

    Bước 5 & 6 — Float Precision Tolerance + Clip [FIX v3]:
        Phép trừ float sinh sai số IEEE 754 (~0.0001 VND). Nếu không lọc,
        hàng nghìn dòng bị gán leakage ảo, làm nhiễu Data Mart và ML features.
        Fix: raw_leakage < config.MATH_TOLERANCE_MIN (1.0 VND) → coi là 0.
        Sau đó clip(lower=0) để chỉ giữ trường hợp công ty bị thiệt.

    Args:
        order_items (pd.DataFrame): Bảng chi tiết đơn hàng.
        promotions (pd.DataFrame): Bảng khuyến mãi.

    Returns:
        pd.DataFrame: Các cột:
            order_id, product_id, promo_id, promo_id_2,
            quantity, line_gross_revenue, discount_amount,
            promo_discount_p1, promo_discount_p2, revenue_leakage.
    """
    df_oi = order_items.copy()
    df_p  = promotions[[
        "promo_id", "promo_type", "discount_value", "min_order_value", "stackable_flag"
    ]].copy()
    df_p["min_order_value"] = df_p["min_order_value"].fillna(0)

    # Bước 1: Tổng giá trị đơn hàng để kiểm tra MOV
    df_oi["line_gross_revenue"] = df_oi["quantity"] * df_oi["unit_price"]
    order_totals = (
        df_oi.groupby("order_id")["line_gross_revenue"]
        .sum()
        .reset_index(name="order_val")
    )
    df_oi = df_oi.merge(order_totals, on="order_id", how="left")

    # Bước 2: Join Promo (mã chính p1_ và mã phụ p2_)
    df_m = df_oi.merge(df_p.add_prefix("p1_"), left_on="promo_id",   right_on="p1_promo_id", how="left")
    if "promo_id_2" in df_m.columns:
        df_m = df_m.merge(df_p.add_prefix("p2_"), left_on="promo_id_2", right_on="p2_promo_id", how="left")
    else:
        df_m["p2_promo_id"] = np.nan

    # Bước 3: Xác định vi phạm
    abuse_mask  = df_m["p2_promo_id"].notna() & (df_m["p1_stackable_flag"] == 0)
    p1_mov_mask = (
        df_m["p1_promo_id"].notna()
        & (df_m["p1_min_order_value"] > 0)       # [FIX v3] bỏ qua min_order_value = 0
        & (df_m["order_val"] < df_m["p1_min_order_value"])
    )
    p2_mov_mask = (
        df_m["p2_promo_id"].notna()
        & (df_m["p2_min_order_value"] > 0)
        & (df_m["order_val"] < df_m["p2_min_order_value"])
    )

    # Bước 4: Tính expected discount theo luật
    d1_fixed = df_m["quantity"] * df_m["p1_discount_value"]
    d1_pct   = df_m["quantity"] * df_m["unit_price"] * (df_m["p1_discount_value"] / 100.0)
    expected_d1 = np.where(df_m["p1_promo_type"] == "fixed", d1_fixed, d1_pct)
    expected_d1 = np.where(p1_mov_mask, 0, expected_d1)

    expected_d2 = np.zeros(len(df_m))
    if "p2_promo_type" in df_m.columns:
        d2_fixed = df_m["quantity"] * df_m["p2_discount_value"]
        d2_pct   = df_m["quantity"] * df_m["unit_price"] * (df_m["p2_discount_value"] / 100.0)
        expected_d2 = np.where(df_m["p2_promo_type"] == "fixed", d2_fixed, d2_pct)
        expected_d2 = np.where(p2_mov_mask | abuse_mask, 0, expected_d2)

    df_m["promo_discount_p1"] = pd.Series(expected_d1).fillna(0).values
    df_m["promo_discount_p2"] = pd.Series(expected_d2).fillna(0).values
    df_m["expected_discount"] = df_m["promo_discount_p1"] + df_m["promo_discount_p2"]

    # Bước 5 & 6: Float tolerance filter + clip
    raw_leakage = df_m["discount_amount"].fillna(0) - df_m["expected_discount"]
    df_m["revenue_leakage"] = np.where(
        raw_leakage.abs() < config.MATH_TOLERANCE_MIN,
        0.0,
        raw_leakage
    )
    df_m["revenue_leakage"] = df_m["revenue_leakage"].clip(lower=0)

    cols_to_keep  = [
        "order_id", "product_id", "promo_id", "promo_id_2",
        "quantity", "line_gross_revenue", "discount_amount",
        "promo_discount_p1", "promo_discount_p2",
        "revenue_leakage"
    ]
    existing_cols = [col for col in cols_to_keep if col in df_m.columns]
    return df_m[existing_cols]


# ==========================================
# LỚP 3: GOM NHÓM THEO CỤM (CLUSTERS)
# ==========================================

def _process_transaction_cluster() -> pd.DataFrame:
    """
    Đọc dữ liệu giao dịch, tính Revenue Leakage và gom nhóm thành chỉ số ngày.

    Quy trình:
    1. Đọc orders, order_items, promotions từ Parquet.
    2. Gọi _engineer_promotional_features() → engineered items.
    3. Cache engineered items ra _temp_engineered_items.parquet.
    4. Gom nhóm: item-level → order-level → daily-level.

    Returns:
        pd.DataFrame: Các cột:
            Date, daily_orders, daily_items_sold, daily_gross_revenue,
            daily_actual_discount, daily_revenue_leakage.

    Side Effects:
        Ghi _temp_engineered_items.parquet vào config.PROCESSED.
    """
    logger.info("Đang xử lý Cụm Giao dịch...")
    orders = pd.read_parquet(config.PROCESSED / "orders.parquet")
    items  = pd.read_parquet(config.PROCESSED / "order_items.parquet")
    promos = pd.read_parquet(config.PROCESSED / "promotions.parquet")

    orders["order_date"] = orders["order_date"].dt.normalize()

    engineered_items = _engineer_promotional_features(items, promos)
    engineered_items.to_parquet(config.PROCESSED / "_temp_engineered_items.parquet", index=False)

    order_level = engineered_items.groupby("order_id").agg(
        total_items           =("quantity",          "sum"),
        order_gross_revenue   =("line_gross_revenue", "sum"),
        order_actual_discount =("discount_amount",    "sum"),
        order_leakage         =("revenue_leakage",    "sum")
    ).reset_index()

    daily_tx = orders.merge(order_level, on="order_id", how="left")
    return (
        daily_tx.groupby("order_date").agg(
            daily_orders          =("order_id",             "nunique"),
            daily_items_sold      =("total_items",           "sum"),
            daily_gross_revenue   =("order_gross_revenue",   "sum"),
            daily_actual_discount =("order_actual_discount", "sum"),
            daily_revenue_leakage =("order_leakage",         "sum")
        )
        .reset_index()
        .rename(columns={"order_date": "Date"})
    )


def _process_customer_cluster() -> pd.DataFrame:
    """
    Gom nhóm dữ liệu hoàn trả (returns) và đánh giá (reviews) thành chỉ số ngày.

    Xử lý 2 nguồn dữ liệu độc lập, merge bằng how='outer' để giữ ngày chỉ có
    returns hoặc chỉ có reviews. avg_daily_rating sẽ còn NaN tại ngày không có
    review — imputation (ffill an toàn, không dùng global median) được thực hiện
    trong build_master_table() sau khi date spine xác định.

    Returns:
        pd.DataFrame: Các cột:
            Date, daily_returns, daily_refund_amount,
            avg_daily_rating (nullable), daily_review_count.
    """
    logger.info("Đang xử lý Cụm Khách hàng...")
    returns = pd.read_parquet(config.PROCESSED / "returns.parquet")
    reviews = pd.read_parquet(config.PROCESSED / "reviews.parquet")

    returns["Date"] = returns["return_date"].dt.normalize()
    daily_ret = returns.groupby("Date").agg(
        daily_returns       =("return_id",    "nunique"),
        daily_refund_amount =("refund_amount", "sum")
    ).reset_index()

    reviews["Date"] = reviews["review_date"].dt.normalize()
    daily_rev = reviews.groupby("Date").agg(
        avg_daily_rating   =("rating",    "mean"),
        daily_review_count =("review_id", "nunique")  # [FIX v4] nunique để nhất quán với daily_returns/daily_orders
    ).reset_index()

    return pd.merge(daily_ret, daily_rev, on="Date", how="outer")


def _process_operation_cluster(date_spine: pd.DataFrame) -> pd.DataFrame:
    """
    Gom nhóm lưu lượng web (traffic) và ánh xạ tồn kho tháng xuống ngày.

    Web Traffic (granularity: ngày):
        Khai thác đầy đủ các cột có trong raw mà phiên bản trước bỏ sót:
          - total_sessions, total_visitors, total_page_views
          - avg_bounce_rate, avg_session_duration_sec
          - pct_organic, pct_paid, pct_social, pct_email, pct_direct, pct_referral:
            Pivot traffic_source → tỷ trọng kênh theo ngày. Cho phép phân tích
            channel mix không cần one-hot encoding riêng ở bước feature engineering.

        Gap 2012 (web_traffic từ 2013-01-01, date_spine từ 2012-07-04):
        ~6 tháng đầu không có traffic data → NaN sau merge, được điền 0 trong
        build_master_table() (không có session thực sự = 0, không phải missing).

    Inventory (granularity: tháng → ngày via Forward + Back Fill):
        Các chỉ số:
          - avg_fill_rate         : Tỷ lệ đáp ứng đơn hàng tháng.
          - monthly_stockout_days : Tổng ngày hết hàng (chỉ số THÁNG, ffill xuống ngày).
          - stockout_rate_per_day : Normalize về tỷ lệ, an toàn hơn khi dùng ML feature.
          - avg_stock_on_hand     : Proxy supply constraint.
          - avg_days_of_supply    : Clip tại 365 ngày — outlier max=68,100 theo profiling.

        [FIX v3 — Missing Back-fill]:
        ffill() đơn thuần để NaN tại đầu chuỗi nếu date_spine bắt đầu trước
        snapshot inventory đầu tiên. Thêm bfill() sau ffill() để điền ngược
        từ record đầu tiên, đảm bảo không có NaN cấu trúc khi train ML.

    Args:
        date_spine (pd.DataFrame): DataFrame chứa cột 'Date' liên tục.

    Returns:
        pd.DataFrame: Bảng vận hành theo ngày.
    """
    logger.info("Đang xử lý Cụm Vận hành...")
    traffic   = pd.read_parquet(config.PROCESSED / "web_traffic.parquet")
    inventory = pd.read_parquet(config.PROCESSED / "inventory.parquet")

    # --- Web Traffic ---
    traffic["Date"] = traffic["date"].dt.normalize()

    # Pivot traffic_source → binary columns, rồi tính mean = tỷ trọng theo ngày
    source_dummies = pd.get_dummies(traffic["traffic_source"], prefix="src")
    traffic = pd.concat([traffic, source_dummies], axis=1)
    src_agg = {col: (col, "mean") for col in source_dummies.columns}

    daily_traffic = (
        traffic.groupby("Date")
        .agg(
            total_sessions           =("sessions",                 "sum"),
            total_visitors           =("unique_visitors",          "sum"),
            total_page_views         =("page_views",               "sum"),
            avg_bounce_rate          =("bounce_rate",              "mean"),
            avg_session_duration_sec =("avg_session_duration_sec", "mean"),
            **src_agg
        )
        .rename(columns=lambda c: c.replace("src_", "pct_"))
        .reset_index()
    )

    # --- Inventory (Monthly → Daily) ---
    inventory["YearMonth"]    = inventory["snapshot_date"].dt.to_period("M")
    inventory["days_in_month"] = inventory["snapshot_date"].dt.days_in_month

    monthly_inv = inventory.groupby("YearMonth").agg(
        avg_fill_rate         =("fill_rate",      "mean"),
        monthly_stockout_days =("stockout_days",  "sum"),
        avg_stock_on_hand     =("stock_on_hand",  "mean"),
        avg_days_of_supply    =("days_of_supply", "mean"),
        days_in_month         =("days_in_month",  "first")
    ).reset_index()

    # Clip outlier: days_of_supply max=68,100 theo profiling → ceiling 365 (1 năm)
    monthly_inv["avg_days_of_supply"] = monthly_inv["avg_days_of_supply"].clip(upper=365)

    # Normalize stockout về tỷ lệ ngày để tránh sai lệch tháng dài/ngắn
    monthly_inv["stockout_rate_per_day"] = (
        monthly_inv["monthly_stockout_days"] / monthly_inv["days_in_month"].clip(lower=1)
    )

    temp_spine = date_spine.copy()
    temp_spine["YearMonth"] = temp_spine["Date"].dt.to_period("M")

    inv_fill_cols = [
        "avg_fill_rate", "monthly_stockout_days", "stockout_rate_per_day",
        "avg_stock_on_hand", "avg_days_of_supply"
    ]

    daily_inv = (
        temp_spine.merge(monthly_inv, on="YearMonth", how="left")
        .drop(columns=["YearMonth", "days_in_month"])
        .sort_values("Date")
    )
    # [FIX v3] ffill() + bfill() để phủ kín cả đầu lẫn cuối chuỗi thời gian
    daily_inv[inv_fill_cols] = daily_inv[inv_fill_cols].ffill().bfill()

    return pd.merge(daily_traffic, daily_inv, on="Date", how="outer")


# ==========================================
# LỚP 4A: XÂY DỰNG MASTER TABLE (CHO MACHINE LEARNING)
# ==========================================

def build_master_table():
    """
    Lắp ráp Master Table cuối cùng bằng cách left-join tất cả cluster vào Date Spine.

    Quy trình:
    1. Date Spine liên tục từ config.DATE_SPINE_START đến config.DATE_SPINE_END.
    2. Merge bảng sales (target: Revenue, COGS) vào Date Spine.
    3. Left-join 3 cluster lần lượt.
    4. Imputation (thứ tự quan trọng):

       a) zero_fill_cols: Điền 0 cho cột đếm/tổng và traffic cols (gap 2012).
          Đây là semantic zeros, không phải missing.

       b) pct_* cols (traffic source): Điền 0 tương tự zero_fill_cols —
          ngày không có traffic → tất cả pct_ = 0.

       c) avg_daily_rating [FIX v3 — Tránh Data Leakage]:
          Phiên bản cũ: fillna(global_median) → median tính trên toàn chuỗi
          (bao gồm dữ liệu 2023-2024) → rò rỉ future information vào tập train,
          mô hình ML học được xu hướng rating tương lai → overfitting nặng.
          Fix: ffill() (chỉ dùng thông tin quá khứ tại mỗi thời điểm),
          sau đó fillna(0) cho các ngày đầu chuỗi chưa có bất kỳ review nào.

       d) daily_net_revenue: Net Revenue = Gross Revenue - Discount - Refund.

    5. Log NaN còn lại để phát hiện vấn đề tiềm ẩn.
    6. Lưu master_table.parquet.

    Side Effects:
        - Gọi _process_transaction_cluster() → _temp_engineered_items.parquet.
        - Ghi master_table.parquet vào config.PROCESSED.
    """
    logger.info("=== Bắt đầu lắp ráp Master Table ===")

    date_spine = pd.DataFrame({
        "Date": pd.date_range(start=config.DATE_SPINE_START, end=config.DATE_SPINE_END, freq="D")
    })

    sales = pd.read_parquet(config.PROCESSED / "sales.parquet")
    sales["Date"] = sales["Date"].dt.normalize()
    master = date_spine.merge(sales, on="Date", how="left")

    master = master.merge(_process_transaction_cluster(), on="Date", how="left")
    master = master.merge(_process_customer_cluster(),    on="Date", how="left")
    master = master.merge(_process_operation_cluster(date_spine), on="Date", how="left")

    # a) Zero-fill: cột đếm/tổng — không phát sinh sự kiện = 0
    zero_fill_cols = [
        "daily_orders", "daily_items_sold", "daily_returns",
        "daily_actual_discount", "daily_revenue_leakage",
        "daily_gross_revenue", "daily_refund_amount", "daily_review_count",
        # Traffic: gap 2012 trước khi web_traffic bắt đầu → 0
        "total_sessions", "total_visitors", "total_page_views",
    ]
    # b) pct_* cols: traffic source proportion → 0 khi không có traffic
    pct_cols = [c for c in master.columns if c.startswith("pct_")]
    zero_fill_cols += pct_cols

    cols_present = [c for c in zero_fill_cols if c in master.columns]
    master[cols_present] = master[cols_present].fillna(0)

    # c) avg_daily_rating: ffill (past-only) → fillna(0) cho đầu chuỗi
    # [FIX v3] Không dùng global median: bao gồm future data → Data Leakage
    master["avg_daily_rating"] = (
        master["avg_daily_rating"]
        .ffill()
        .fillna(0)
    )

    # d) Net Revenue
    master["daily_net_revenue"] = (
        master["daily_gross_revenue"]
        - master["daily_actual_discount"]
        - master["daily_refund_amount"]
    )

    # Kiểm tra NaN còn lại
    nan_summary = master.isnull().sum()
    nan_remaining = nan_summary[nan_summary > 0]
    if not nan_remaining.empty:
        logger.warning(f"NaN còn lại trong Master Table:\n{nan_remaining.to_string()}")
    else:
        logger.info("   Master Table: Không có NaN còn lại.")

    master.to_parquet(config.PROCESSED / "master_table.parquet", index=False)
    logger.info(f"=== Đã lưu Master Table: {master.shape} ===")


# ==========================================
# LỚP 4B: XÂY DỰNG DATA MARTS (CHO EDA & BI)
# ==========================================

def _build_customer_cohort_mart():
    """
    Xây dựng Customer Cohort Mart — phân tích hành vi và CLV theo customer_id.

    3 nhóm chỉ số:
    1. Cohort   : first_order_date → cohort tháng/năm.
    2. Retention: is_retained_1m, is_retained_3m (flag days > 0 để loại first order).
       Business rule: days = 0 (first order) bị loại — retention đo hành vi QUAY LẠI.
       Khách chỉ mua 1 lần có is_retained_1m = is_retained_3m = 0 (churn đúng nghĩa).
    3. CLV      : total_revenue, revenue_per_order.

    Raises:
        FileNotFoundError: Nếu _temp_engineered_items.parquet chưa tồn tại.

    Side Effects:
        Ghi mart_customer_cohort.parquet vào config.PROCESSED.
    """
    logger.info("   -> Xây dựng Customer Cohort Mart...")

    temp_path = config.PROCESSED / "_temp_engineered_items.parquet"
    if not temp_path.exists():
        raise FileNotFoundError(
            "_temp_engineered_items.parquet chưa tồn tại. "
            "Hãy chạy build_master_table() trước."
        )

    orders    = pd.read_parquet(config.PROCESSED / "orders.parquet")
    items     = pd.read_parquet(temp_path)
    customers = pd.read_parquet(config.PROCESSED / "customers.parquet")

    first_orders   = orders.groupby("customer_id")["order_date"].min().reset_index(name="first_order_date")
    customer_stats = orders.merge(first_orders, on="customer_id", how="left")
    customer_stats["days_since_first_order"] = (
        customer_stats["order_date"] - customer_stats["first_order_date"]
    ).dt.days

    customer_stats["is_retained_1m"] = (
        (customer_stats["days_since_first_order"] > 0) &
        (customer_stats["days_since_first_order"] <= 30)
    ).astype(int)
    customer_stats["is_retained_3m"] = (
        (customer_stats["days_since_first_order"] > 0) &
        (customer_stats["days_since_first_order"] <= 90)
    ).astype(int)

    retention_data = customer_stats.groupby("customer_id").agg(
        is_retained_1m =("is_retained_1m", "max"),
        is_retained_3m =("is_retained_3m", "max"),
        total_orders   =("order_id",        "nunique")
    ).reset_index()

    item_revenue     = items.groupby("order_id")["line_gross_revenue"].sum().reset_index(name="order_revenue")
    order_rev_merged = orders.merge(item_revenue, on="order_id", how="left")
    clv_data         = (
        order_rev_merged.groupby("customer_id")["order_revenue"]
        .sum()
        .reset_index(name="total_revenue")
    )

    customer_mart = customers.merge(first_orders,    on="customer_id", how="left")
    customer_mart = customer_mart.merge(retention_data, on="customer_id", how="left")
    customer_mart = customer_mart.merge(clv_data,       on="customer_id", how="left")
    customer_mart["revenue_per_order"] = (
        customer_mart["total_revenue"] / customer_mart["total_orders"].clip(lower=1)
    )

    customer_mart.to_parquet(config.PROCESSED / "mart_customer_cohort.parquet", index=False)


def _build_product_performance_mart():
    """
    Xây dựng Product Performance Mart — đánh giá hiệu suất theo product_id.

    4 nhóm chỉ số:
    1. Sales  : units_sold, total_gross_revenue, total_discount, total_leakage.
    2. Returns: units_returned, total_refund_amount.
    3. Net Revenue = gross - discount - refund.
    4. Margin : return_rate, total_cogs_amount, gross_margin_rate.
       gross_margin_rate chia cho gross_revenue (không phải net) để tránh kết quả
       vô nghĩa khi net_revenue âm.

    Raises:
        FileNotFoundError: Nếu _temp_engineered_items.parquet chưa tồn tại.

    Side Effects:
        Ghi mart_product_performance.parquet vào config.PROCESSED.
    """
    logger.info("   -> Xây dựng Product Performance Mart...")

    temp_path = config.PROCESSED / "_temp_engineered_items.parquet"
    if not temp_path.exists():
        raise FileNotFoundError(
            "_temp_engineered_items.parquet chưa tồn tại. "
            "Hãy chạy build_master_table() trước."
        )

    items    = pd.read_parquet(temp_path)
    products = pd.read_parquet(config.PROCESSED / "products.parquet")
    returns  = pd.read_parquet(config.PROCESSED / "returns.parquet")

    product_sales = items.groupby("product_id").agg(
        units_sold          =("quantity",          "sum"),
        total_gross_revenue =("line_gross_revenue", "sum"),
        total_discount      =("discount_amount",    "sum"),
        total_leakage       =("revenue_leakage",    "sum")
    ).reset_index()

    product_returns = returns.groupby("product_id").agg(
        units_returned      =("return_quantity", "sum"),
        total_refund_amount =("refund_amount",   "sum")
    ).reset_index()

    product_mart = products.merge(product_sales,      on="product_id", how="left")
    product_mart = product_mart.merge(product_returns, on="product_id", how="left").fillna(0)

    product_mart["net_revenue"]       = (
        product_mart["total_gross_revenue"]
        - product_mart["total_discount"]
        - product_mart["total_refund_amount"]
    )
    product_mart["return_rate"]       = (
        product_mart["units_returned"] / product_mart["units_sold"].clip(lower=1)
    )
    product_mart["total_cogs_amount"] = product_mart["units_sold"] * product_mart["cogs"]
    # [FIX v4] Gross Margin = (Gross Revenue - COGS) / Gross Revenue
    # Phiên bản cũ dùng net_revenue (đã trừ discount + refund) → tính ra Net Margin,
    # không phải Gross Margin. Khi refund lớn, net_revenue âm → gross_margin_rate âm
    # dù sản phẩm vẫn có lãi gộp thực → sai lệch nghiêm trọng trong phân tích BI.
    product_mart["gross_margin_rate"] = (
        (product_mart["total_gross_revenue"] - product_mart["total_cogs_amount"])
        / product_mart["total_gross_revenue"].clip(lower=0.01)
    )

    product_mart.to_parquet(config.PROCESSED / "mart_product_performance.parquet", index=False)


def _build_promo_campaign_mart():
    """
    Xây dựng Promo Campaign Mart — đánh giá hiệu quả và ROI theo promo_id (mã chính).

    Chỉ số:
    - Reach   : total_orders_used.
    - Revenue : gross_revenue_generated.
    - Cost    : total_discount_given (dòng tiền thực),
                total_p1_discount (chi phí riêng p1 theo luật nghiệp vụ).
    - Leakage : total_leakage_caused.

    Efficiency:
    - actual_discount_depth  : total_discount_given / gross_revenue.
    - net_revenue_after_disc : gross_revenue - total_p1_discount.
    - promo_roi              [FIX v3 — ROI Exploding]:
        Cũ: / total_p1_discount.clip(lower=0.01) → ROI = revenue/0.01 = hàng tỷ
        khi promo không phát sinh chi phí (toàn bộ vi phạm MOV).
        Fix: np.where(cost <= 0, 0.0, revenue/cost) → không có chi phí = ROI không đo được = 0.

    Limitation:
        gross_revenue_generated bao gồm doanh thu cả đơn có promo_id_2.
        Không thể tách phần thuần của p1 ở granularity order-item level.

    Raises:
        FileNotFoundError: Nếu _temp_engineered_items.parquet chưa tồn tại.

    Side Effects:
        Ghi mart_promo_campaign.parquet vào config.PROCESSED.
    """
    logger.info("   -> Xây dựng Promo Campaign Mart...")

    temp_path = config.PROCESSED / "_temp_engineered_items.parquet"
    if not temp_path.exists():
        raise FileNotFoundError(
            "_temp_engineered_items.parquet chưa tồn tại. "
            "Hãy chạy build_master_table() trước."
        )

    promos = pd.read_parquet(config.PROCESSED / "promotions.parquet")
    items  = pd.read_parquet(temp_path)

    promo_items = items[items["promo_id"].notna()]
    promo_perf  = promo_items.groupby("promo_id").agg(
        total_orders_used       =("order_id",           "nunique"),
        gross_revenue_generated =("line_gross_revenue", "sum"),
        total_discount_given    =("discount_amount",    "sum"),
        total_p1_discount       =("promo_discount_p1",  "sum"),
        total_leakage_caused    =("revenue_leakage",    "sum")
    ).reset_index()

    promo_mart = promos.merge(promo_perf, on="promo_id", how="left").fillna(0)

    promo_mart["actual_discount_depth"]  = (
        promo_mart["total_discount_given"]
        / promo_mart["gross_revenue_generated"].clip(lower=0.01)
    )
    promo_mart["net_revenue_after_disc"] = (
        promo_mart["gross_revenue_generated"] - promo_mart["total_p1_discount"]
    )
    # [FIX v3] np.where tránh ROI vô cực khi chi phí = 0
    promo_mart["promo_roi"] = np.where(
        promo_mart["total_p1_discount"] <= 0,
        0.0,
        promo_mart["net_revenue_after_disc"] / promo_mart["total_p1_discount"]
    )

    promo_mart.to_parquet(config.PROCESSED / "mart_promo_campaign.parquet", index=False)


def build_all_data_marts():
    """
    Orchestrator xuất toàn bộ Data Mart chuyên đề phục vụ EDA và BI.

    Thứ tự:
      1. _build_customer_cohort_mart()     → mart_customer_cohort.parquet
      2. _build_product_performance_mart() → mart_product_performance.parquet
      3. _build_promo_campaign_mart()      → mart_promo_campaign.parquet

    Raises:
        FileNotFoundError: Nếu build_master_table() chưa chạy trước.
    """
    logger.info("=== Khởi tạo hệ thống Data Marts (Dành cho EDA) ===")
    _build_customer_cohort_mart()
    _build_product_performance_mart()
    _build_promo_campaign_mart()
    logger.info("=== Hệ thống Data Marts đã sẵn sàng ===")


def run_prep(force_rebuild: bool = False):
    """
    Entry point chính — thực thi toàn bộ Data Preprocessing Pipeline.

    - force_rebuild=False (default): Bỏ qua nếu master_table.parquet đã tồn tại.
    - force_rebuild=True: Chạy lại toàn bộ pipeline.

    Thứ tự khi chạy:
      1. initialize_parquet_storage() — CSV → Parquet.
      2. build_master_table()         — Master Table cho ML.
      3. build_all_data_marts()       — Data Marts cho EDA.

    Args:
        force_rebuild (bool): Mặc định False để tránh re-run vô ý trong notebook.
    """
    master_path = config.PROCESSED / "master_table.parquet"

    if force_rebuild or not master_path.exists():
        initialize_parquet_storage(config.SOURCE_TABLES)
        build_master_table()
        build_all_data_marts()
    else:
        logger.info(
            "Dữ liệu đã sẵn sàng. Bỏ qua để tiết kiệm thời gian "
            "(đặt force_rebuild=True để chạy lại)."
        )