"""
src/feature_eng.py — Agent 2: Feature Engineering.

Pipeline: Load base_table (4,381 ngày) → Build Nhóm A (Calendar/Holiday/Fourier,
không shift) → Build Promotion Intensity (dùng promotions đã được mở rộng sang
2023–2024 bằng chu kỳ 2 năm) → Drop observed daily cols → Split → Ghi
train_features.parquet + test_features.parquet.

THAY ĐỔI SO VỚI PHIÊN BẢN CŨ:
  [FIX 1] load_promotions(): mở rộng chu kỳ khuyến mãi 2021→2023, 2022→2024
          bằng pd.DateOffset(years=2) — đảm bảo tập Test có promo signal.
  [FIX 2] Nhóm B (build_lag_features, build_macd_features,
          build_price_discount_elasticity, build_web_traffic_features,
          build_inventory_features) bị VÔ HIỆU HÓA hoàn toàn vì gây Data
          Leakage và NaN toàn bộ khi Horizon > 1 ngày.
          split_and_save() drop thêm OBSERVED_COLS (daily aggregates từ
          data_prep) trước khi ghi file, triệt tiêu mọi rò rỉ.

Tuân thủ PART_II: không print, không for-loop trên DF, không hardcode tham số,
type hint + docstring đầy đủ, không downcast target.
"""

import argparse
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TARGET_COLS: List[str] = ["revenue", "cogs", "margin"]

# Các cột sinh ra từ daily aggregates của data_prep (Agent 1).
# Đây là dữ liệu QUAN SÁT THEO NGÀY — không tồn tại ở tập Test thực tế.
# Phải drop trước khi split để triệt tiêu 100% leakage.
OBSERVED_COLS: List[str] = [
    "year",
    "gross_revenue",
    "total_discount",
    "n_orders",
    "n_unique_customers",
    "n_unique_products",
    "cancelled_rate",
    "n_returns",
    "total_refund",
    "avg_rating",
    "n_reviews",
    "avg_session_duration_sec",
    "n_items_sold",
    "avg_order_value",
    "sessions",
    "page_views",
    "unique_visitors",
    "bounce_rate",
    "avg_fill_rate",
    "avg_stockout_days",
    "pct_stockout_skus",
    "avg_sell_through",
    "unit_price",
    "quantity",
    "discount_amount",
    "avg_delivery_days"
]

# Ngày mùng 1 Tết Nguyên Đán (Dương lịch) hardcode cho 2012–2024.
TET_NEW_YEAR_DATES: List[str] = [
    "2012-01-23",  # Nhâm Thìn
    "2013-02-10",  # Quý Tỵ
    "2014-01-31",  # Giáp Ngọ
    "2015-02-19",  # Ất Mùi
    "2016-02-08",  # Bính Thân
    "2017-01-28",  # Đinh Dậu
    "2018-02-16",  # Mậu Tuất
    "2019-02-05",  # Kỷ Hợi
    "2020-01-25",  # Canh Tý
    "2021-02-12",  # Tân Sửu
    "2022-02-01",  # Nhâm Dần
    "2023-01-22",  # Quý Mão
    "2024-02-10",  # Giáp Thìn
]

# Prefix nhận diện Nhóm A — dùng trong validate
GROUP_A_PREFIXES: Tuple[str, ...] = (
    "day_", "week_", "month", "quarter", "year",
    "is_", "days_to_", "days_from_", "sin_", "cos_",
    "n_active_promos", "max_discount_pct", "has_stackable_promo", "pis_score",
)


# ---------------------------------------------------------------------------
# Parse arguments & Config
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments cho feature_eng.py."""
    parser = argparse.ArgumentParser(
        description="Feature Engineering — tạo train_features.parquet + test_features.parquet"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Đường dẫn tới file config.yaml",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """
    Đọc toàn bộ cấu hình từ config.yaml.

    Parameters
    ----------
    path : Path
        Đường dẫn tới config.yaml.

    Returns
    -------
    dict
        Cấu hình pipeline.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict) -> None:
    """
    Cấu hình logging toàn bộ pipeline từ config.

    Parameters
    ----------
    config : dict
        Cấu hình đọc từ config.yaml.
    """
    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)

    from datetime import datetime
    log_file = log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"

    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


# ---------------------------------------------------------------------------
# Quy Tắc 7: Downcast — ngoại trừ target
# ---------------------------------------------------------------------------

def downcast_df(df: pd.DataFrame, exclude_cols: List[str]) -> pd.DataFrame:
    """
    Downcast dtype để tiết kiệm RAM. Bỏ qua các cột trong exclude_cols.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cần downcast.
    exclude_cols : List[str]
        Cột KHÔNG được downcast (target + date).

    Returns
    -------
    pd.DataFrame
        DataFrame sau khi downcast.
    """
    for col in df.columns:
        if col in exclude_cols:
            continue
        if df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_base_table(processed_dir: Path) -> pd.DataFrame:
    """
    Đọc base_table.parquet (output của Agent 1).
    Kỳ vọng 4,381 dòng: 3,833 dòng train (có dữ liệu) + 548 dòng test (NaN).

    Parameters
    ----------
    processed_dir : Path
        Thư mục chứa file parquet đã xử lý.

    Returns
    -------
    pd.DataFrame
        DataFrame đã sắp xếp theo date tăng dần.
    """
    path = processed_dir / "base_table.parquet"
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    n_nan_rev = int(df["revenue"].isna().sum()) if "revenue" in df.columns else -1
    logger.info(
        "Đã load base_table.parquet: %d dòng × %d cột | revenue NaN: %d",
        df.shape[0], df.shape[1], n_nan_rev,
    )
    return df


def load_promotions(raw_dir: Path) -> pd.DataFrame:
    """
    Đọc, chuẩn hóa và mở rộng chu kỳ promotions.csv sang giai đoạn dự báo.

    Promotions.csv chỉ chứa dữ liệu đến hết 2022. Hàm này lọc riêng các
    sự kiện năm 2021 và 2022, tịnh tiến thời gian lên 2 năm bằng
    pd.DateOffset(years=2) (2021→2023, 2022→2024), cập nhật tên event
    tương ứng, rồi concat vào bảng gốc. Nhờ đó build_promotion_intensity()
    sẽ có promo signal đầy đủ cho toàn bộ tập Test (2023–2024).

    Quy luật chu kỳ được giữ nguyên:
      - Rural Special: năm lẻ (2021 → 2023)
      - Urban Blowout: năm chẵn (2022 → 2024)
      - Sale tháng 3, 6, 8, 11: lặp lại mỗi năm

    Parameters
    ----------
    raw_dir : Path
        Thư mục chứa file CSV thô.

    Returns
    -------
    pd.DataFrame
        DataFrame promotions với start_date, end_date là datetime64.
        Đã bổ sung các dòng tương ứng năm 2023 và 2024.
        applicable_category giữ NaN (NULLABLE_BY_DESIGN).
    """
    df = pd.read_csv(
        raw_dir / "promotions.csv",
        parse_dates=["start_date", "end_date"],
    )
    df.columns = [c.lower() for c in df.columns]
    df["promo_type"]     = df["promo_type"].astype("category")
    df["stackable_flag"] = pd.to_numeric(df["stackable_flag"], downcast="integer")
    df["discount_value"] = pd.to_numeric(df["discount_value"], downcast="float")
    # applicable_category là NULLABLE_BY_DESIGN — không impute
    if "applicable_category" in df.columns:
        df["applicable_category"] = df["applicable_category"].astype("category")

    logger.info("Đã load promotions.csv gốc: %d dòng", len(df))

    # --- Mở rộng chu kỳ: 2021 → 2023, 2022 → 2024 ---
    offset = pd.DateOffset(years=2)

    # Lọc riêng từng năm nguồn
    mask_2021 = df["start_date"].dt.year == 2021
    mask_2022 = df["start_date"].dt.year == 2022

    extended_parts: List[pd.DataFrame] = []

    for mask in (mask_2021, mask_2022):
        if not mask.any():
            continue

        chunk = df[mask].copy()

        # Tịnh tiến start_date và end_date lên 2 năm — vectorized
        chunk["start_date"] = chunk["start_date"] + offset
        chunk["end_date"]   = chunk["end_date"]   + offset

        # Cập nhật tên event: thay thế chuỗi năm nguồn bằng năm đích
        # (VD: "Rural Special 2021" → "Rural Special 2023")
        src_year  = str(chunk["start_date"].dt.year.iloc[0] - 2)  # năm nguồn
        dest_year = str(chunk["start_date"].dt.year.iloc[0])       # năm đích
        if "event_name" in chunk.columns:
            chunk["event_name"] = chunk["event_name"].str.replace(
                src_year, dest_year, regex=False
            )

        extended_parts.append(chunk)

    if extended_parts:
        df_extended = pd.concat([df] + extended_parts, ignore_index=True)
        df_extended = df_extended.sort_values("start_date").reset_index(drop=True)
        logger.info(
            "Promotions sau khi mở rộng chu kỳ: %d dòng (thêm %d dòng cho 2023–2024)",
            len(df_extended), len(df_extended) - len(df),
        )
        return df_extended

    logger.warning(
        "Không tìm thấy promo năm 2021 hoặc 2022 — trả về bảng gốc không mở rộng"
    )
    return df


# ---------------------------------------------------------------------------
# Nhóm A — Date-based Features (KHÔNG shift)
# ---------------------------------------------------------------------------

def build_calendar_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    [Nhóm A] Thêm calendar features từ cột date.
    Tính trên TOÀN BỘ 4,381 ngày — không có NaN, không shift.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã có cột date (datetime64), sorted by date.
    cfg : dict
        Cấu hình pipeline.

    Returns
    -------
    pd.DataFrame
        DataFrame với các cột calendar được thêm vào.
    """
    if not cfg["features"]["base_time_features"]["enabled"]:
        logger.info("base_time_features: DISABLED — bỏ qua")
        return df

    cols_before = df.shape[1]
    dt = df["date"].dt

    df["day_of_week"]      = dt.dayofweek.astype("int8")
    df["day_of_month"]     = dt.day.astype("int8")
    df["week_of_year"]     = dt.isocalendar().week.astype("int8")
    df["month"]            = dt.month.astype("int8")
    df["quarter"]          = dt.quarter.astype("int8")
    df["year"]             = dt.year.astype("int16")
    df["is_weekend"]       = (dt.dayofweek >= 5).astype("int8")
    df["is_month_start"]   = dt.is_month_start.astype("int8")
    df["is_month_end"]     = dt.is_month_end.astype("int8")
    df["is_quarter_start"] = dt.is_quarter_start.astype("int8")
    df["is_quarter_end"]   = dt.is_quarter_end.astype("int8")

    # FIX-1: linear trend proxy for tree-based models
    df["trend_index"] = (df["date"] - df["date"].min()).dt.days.astype(int)

    cols_added = df.shape[1] - cols_before
    logger.info("[Nhóm A] build_calendar_features: +%d cột", cols_added)
    return df


def build_vn_holidays(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    [Nhóm A] Thêm các feature ngày lễ Việt Nam.
    Tính trên TOÀN BỘ 4,381 ngày — không có NaN, không shift.
    Gồm: is_tet (±7 ngày), is_30_4, is_1_5, is_2_9, is_christmas,
    days_to_next_holiday, days_from_last_holiday.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã có cột date (datetime64).
    cfg : dict
        Cấu hình pipeline.

    Returns
    -------
    pd.DataFrame
        DataFrame với các cột holiday được thêm vào.
    """
    if not cfg["features"]["vn_holidays"]["enabled"]:
        logger.info("vn_holidays: DISABLED — bỏ qua")
        return df

    cols_before = df.shape[1]

    # is_tet: broadcasting (N × M) để tìm khoảng cách tới mỗi Tết
    tet_dates = pd.to_datetime(TET_NEW_YEAR_DATES)
    date_vals = df["date"].values.astype("datetime64[D]")    # (N,)
    tet_vals  = tet_dates.values.astype("datetime64[D]")     # (M,)

    diff_matrix = np.abs(
        date_vals[:, None].astype("int64") - tet_vals[None, :].astype("int64")
    )  # (N, M), đơn vị ngày
    min_days_to_tet = diff_matrix.min(axis=1)
    df["is_tet"] = (min_days_to_tet <= 7).astype("int8")

    # Ngày lễ cố định — vectorized
    month_v = df["date"].dt.month.values
    day_v   = df["date"].dt.day.values

    df["is_30_4"]      = ((month_v == 4)  & (day_v == 30)).astype("int8")
    df["is_1_5"]       = ((month_v == 5)  & (day_v == 1)).astype("int8")
    df["is_2_9"]       = ((month_v == 9)  & (day_v == 2)).astype("int8")
    df["is_christmas"] = ((month_v == 12) & (day_v == 25)).astype("int8")

    # Gộp tất cả holiday cho mọi năm có mặt trong df
    years = df["date"].dt.year.unique()
    fixed_list: List[pd.Timestamp] = []
    for yr in years:
        fixed_list.extend([
            pd.Timestamp(yr, 4, 30),
            pd.Timestamp(yr, 5, 1),
            pd.Timestamp(yr, 9, 2),
            pd.Timestamp(yr, 12, 25),
        ])

    all_holidays = np.sort(np.unique(np.concatenate([
        tet_vals,
        np.array(fixed_list, dtype="datetime64[D]"),
    ])))

    date_int = date_vals.astype("int64")
    holi_int = all_holidays.astype("int64")

    # days_to_next_holiday — np.searchsorted
    idx_next = np.searchsorted(holi_int, date_int, side="left")
    idx_next_clipped = np.clip(idx_next, 0, len(holi_int) - 1)
    days_to_next = holi_int[idx_next_clipped] - date_int
    days_to_next = np.where(idx_next >= len(holi_int), 999, days_to_next)
    df["days_to_next_holiday"] = days_to_next.astype("int16")

    # days_from_last_holiday
    idx_prev = np.searchsorted(holi_int, date_int, side="left") - 1
    idx_prev_clipped = np.clip(idx_prev, 0, len(holi_int) - 1)
    days_from_last = date_int - holi_int[idx_prev_clipped]
    days_from_last = np.where(idx_prev < 0, 999, days_from_last)
    df["days_from_last_holiday"] = days_from_last.astype("int16")

    cols_added = df.shape[1] - cols_before
    logger.info("[Nhóm A] build_vn_holidays: +%d cột", cols_added)
    return df


def build_fourier_seasonality(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    [Nhóm A] Thêm Fourier seasonality features: sin/cos cho các period 7, 30, 365.
    Tính trên TOÀN BỘ 4,381 ngày — không có NaN, không shift.
    Vectorized hoàn toàn bằng numpy.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã có cột date (datetime64).
    cfg : dict
        Cấu hình pipeline.

    Returns
    -------
    pd.DataFrame
        DataFrame với sin/cos features được thêm vào.
    """
    if not cfg["features"]["fourier_seasonality"]["enabled"]:
        logger.info("fourier_seasonality: DISABLED — bỏ qua")
        return df

    cols_before = df.shape[1]
    periods: List[int] = cfg["features"]["fourier_seasonality"]["periods"]
    n_terms: int       = cfg["features"]["fourier_seasonality"]["n_terms"]

    t = (df["date"] - df["date"].min()).dt.days.values.astype("float64")

    for p in periods:
        for k in range(1, n_terms + 1):
            angle = 2.0 * np.pi * k * t / p
            df[f"sin_{p}_{k}"] = np.sin(angle).astype("float32")
            df[f"cos_{p}_{k}"] = np.cos(angle).astype("float32")

    cols_added = df.shape[1] - cols_before
    logger.info(
        "[Nhóm A] build_fourier_seasonality: +%d cột (periods=%s, n_terms=%d)",
        cols_added, periods, n_terms,
    )
    return df


# ---------------------------------------------------------------------------
# Nhóm B — ĐÃ BỊ VÔ HIỆU HÓA
# ---------------------------------------------------------------------------
# Lý do: Horizon dự báo = 1.5 năm (548 ngày). Với horizon > 1 ngày:
#   - shift(1) tại tập Test → NaN từ ngày thứ 2 trở đi → model không có input.
#   - Các cột nguồn (gross_revenue, sessions, ...) là daily observables —
#     không tồn tại ở tập Test thực tế → nếu giữ lại là 100% leakage.
# Giải pháp: chỉ dùng Nhóm A (pure date-based) + Promotion Intensity
# (được tái tạo từ quy luật chu kỳ, không phải observational data).
#
# Các hàm dưới đây được giữ lại trong codebase với trạng thái DISABLED
# để tham khảo và có thể kích hoạt lại khi chuyển sang bài toán 1-step-ahead.
# ---------------------------------------------------------------------------

def build_lag_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    [Nhóm B — DISABLED] Lag và rolling features trên revenue/cogs.

    VÔ HIỆU HÓA vĩnh viễn cho bài toán Horizon 1.5 năm.
    shift(1) chỉ hợp lệ cho dự báo 1 ngày. Với Horizon > 1 ngày,
    tập Test sẽ nhận NaN toàn bộ từ ngày thứ 2 trở đi.
    Ngoài ra revenue/cogs là observed cols → leakage nếu còn trong Test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame pipeline.
    cfg : dict
        Cấu hình pipeline (không dùng).

    Returns
    -------
    pd.DataFrame
        DataFrame không thay đổi.
    """
    logger.info(
        "[Nhóm B] build_lag_features: DISABLED — "
        "không tương thích với Horizon 1.5 năm (leakage + NaN toàn bộ Test)"
    )
    return df


def build_macd_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    [Nhóm B — DISABLED] MACD features trên lag revenue.

    VÔ HIỆU HÓA vĩnh viễn cho bài toán Horizon 1.5 năm.
    MACD dựa trên revenue là observed col → leakage 100% vào Test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame pipeline.
    cfg : dict
        Cấu hình pipeline (không dùng).

    Returns
    -------
    pd.DataFrame
        DataFrame không thay đổi.
    """
    logger.info(
        "[Nhóm B] build_macd_features: DISABLED — "
        "không tương thích với Horizon 1.5 năm (leakage + NaN toàn bộ Test)"
    )
    return df


def build_price_discount_elasticity(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    [Nhóm B — DISABLED] Price Discount Elasticity features.

    VÔ HIỆU HÓA vĩnh viễn cho bài toán Horizon 1.5 năm.
    gross_revenue và total_discount là observed cols → leakage 100% vào Test.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame pipeline.
    cfg : dict
        Cấu hình pipeline (không dùng).

    Returns
    -------
    pd.DataFrame
        DataFrame không thay đổi.
    """
    logger.info(
        "[Nhóm B] build_price_discount_elasticity: DISABLED — "
        "không tương thích với Horizon 1.5 năm (leakage + NaN toàn bộ Test)"
    )
    return df


def build_web_traffic_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    [Nhóm B — DISABLED] Web traffic lag features.

    VÔ HIỆU HÓA vĩnh viễn cho bài toán Horizon 1.5 năm.
    sessions, page_views, unique_visitors là observed cols → leakage 100%.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame pipeline.
    cfg : dict
        Cấu hình pipeline (không dùng).

    Returns
    -------
    pd.DataFrame
        DataFrame không thay đổi.
    """
    logger.info(
        "[Nhóm B] build_web_traffic_features: DISABLED — "
        "không tương thích với Horizon 1.5 năm (leakage + NaN toàn bộ Test)"
    )
    return df


def build_inventory_features(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    [Nhóm B — DISABLED] Inventory lag features.

    VÔ HIỆU HÓA vĩnh viễn cho bài toán Horizon 1.5 năm.
    avg_fill_rate, avg_stockout_days... là observed cols → leakage 100%.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame pipeline.
    cfg : dict
        Cấu hình pipeline (không dùng).

    Returns
    -------
    pd.DataFrame
        DataFrame không thay đổi.
    """
    logger.info(
        "[Nhóm B] build_inventory_features: DISABLED — "
        "không tương thích với Horizon 1.5 năm (leakage + NaN toàn bộ Test)"
    )
    return df


# ---------------------------------------------------------------------------
# Promotion Intensity — Nhóm A* (không shift, không leakage sau FIX 1)
# ---------------------------------------------------------------------------

def build_promotion_intensity(
    df: pd.DataFrame,
    promotions_df: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    [Nhóm A*] Tính Promotion Intensity Score (PIS) theo ngày trên 4,381 ngày.

    Sau FIX 1, promotions_df đã được mở rộng sang 2023–2024 bằng chu kỳ 2 năm,
    nên n_active_promos và pis_score sẽ có giá trị đầy đủ cho toàn bộ tập Test.
    Không cần shift vì đây là thông tin lịch khuyến mãi (được lên kế hoạch trước),
    không phải dữ liệu quan sát theo ngày.

    Vectorized bằng broadcasting (N × M) — không có for-loop trên DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã có cột date (4,381 ngày).
    promotions_df : pd.DataFrame
        DataFrame promotions đã chuẩn hóa và mở rộng chu kỳ.
    cfg : dict
        Cấu hình pipeline.

    Returns
    -------
    pd.DataFrame
        DataFrame với n_active_promos, max_discount_pct,
        has_stackable_promo, pis_score được thêm vào.
    """
    if not cfg["features"]["promotion_intensity"]["enabled"]:
        logger.info("promotion_intensity: DISABLED — bỏ qua")
        return df

    cols_before = df.shape[1]

    date_arr  = df["date"].values.astype("datetime64[D]")                    # (N,)
    start_arr = promotions_df["start_date"].values.astype("datetime64[D]")  # (M,)
    end_arr   = promotions_df["end_date"].values.astype("datetime64[D]")    # (M,)

    # active_matrix[i, j] = True nếu date[i] nằm trong khoảng promo[j]
    active_matrix = (
        (date_arr[:, None] >= start_arr[None, :]) &
        (date_arr[:, None] <= end_arr[None, :])
    )  # (N, M)

    df["n_active_promos"] = active_matrix.sum(axis=1).astype("int8")

    # max_discount_pct: chỉ promo loại percentage
    is_pct      = (promotions_df["promo_type"].astype(str) == "percentage").values
    disc_vals   = promotions_df["discount_value"].values.astype("float32")
    disc_matrix = active_matrix * is_pct[None, :] * disc_vals[None, :]
    df["max_discount_pct"] = disc_matrix.max(axis=1).astype("float32")

    # has_stackable_promo
    stackable_vals   = promotions_df["stackable_flag"].values.astype(bool)
    stackable_active = active_matrix & stackable_vals[None, :]
    df["has_stackable_promo"] = stackable_active.any(axis=1).astype("int8")

    # pis_score = n_active × max_discount_pct
    df["pis_score"] = df["n_active_promos"].astype("float32") * df["max_discount_pct"]

    # Kiểm tra: tập Test (2023–2024) phải có ít nhất một ngày có promo
    test_mask = df["date"] >= "2023-01-01"
    n_promo_days_test = int((df.loc[test_mask, "n_active_promos"] > 0).sum())
    logger.info(
        "[Nhóm A*] build_promotion_intensity: +%d cột | "
        "số ngày có promo trong Test (2023–2024): %d",
        df.shape[1] - cols_before,
        n_promo_days_test,
    )
    if n_promo_days_test == 0:
        logger.warning(
            "CẢNH BÁO: Không có ngày nào có promo trong Test — "
            "kiểm tra lại load_promotions() và chu kỳ mở rộng"
        )
    return df


# ---------------------------------------------------------------------------
# Split & Save
# ---------------------------------------------------------------------------

def split_and_save(
    df: pd.DataFrame,
    cfg: dict,
    processed_dir: Path,
) -> None:
    """
    Tách DataFrame 4,381 ngày thành train (≤2022-12-31) và test (≥2023-01-01).

    Trước khi split, drop OBSERVED_COLS — các cột daily aggregates từ data_prep
    (gross_revenue, sessions, page_views, ...) không tồn tại ở tập Test thực tế.
    Giữ các cột này trong train/test sẽ gây Data Leakage 100% vì chúng tương quan
    trực tiếp với target (revenue, cogs, margin).

    Sau khi split, drop thêm TARGET_COLS khỏi test để triệt tiêu leakage target.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame full 4,381 ngày đã có tất cả features.
    cfg : dict
        Cấu hình pipeline.
    processed_dir : Path
        Thư mục output.

    Returns
    -------
    None
    """
    train_end: str  = cfg["data"]["train_end"]   # "2022-12-31"
    test_start: str = cfg["data"]["test_start"]  # "2023-01-01"

    # Bước 1 — Drop OBSERVED_COLS trước khi split để triệt tiêu leakage
    # Các cột này là daily observables: không tồn tại ở Test thực tế,
    # và tương quan trực tiếp với target → leakage 100% nếu còn trong tập Train.
    cols_before_drop = df.shape[1]
    df = df.drop(columns=OBSERVED_COLS, errors="ignore")
    cols_dropped = cols_before_drop - df.shape[1]
    logger.info(
        "Drop OBSERVED_COLS: %d cột bị xóa %s",
        cols_dropped,
        [c for c in OBSERVED_COLS if c in df.columns or True],
    )

    # Bước 2 — Tách train / test theo date
    train_df = df[df["date"] <= train_end].copy()
    test_df  = df[df["date"] >= test_start].copy()

    # Bước 3 — Drop target khỏi test
    test_df = test_df.drop(columns=TARGET_COLS, errors="ignore")

    # Bước 4 — Validate shape
    assert len(train_df) + len(test_df) == len(df), (
        f"Mất dòng khi split: {len(train_df)} + {len(test_df)} != {len(df)}"
    )
    assert len(train_df) == 3833, (
        f"Train phải có 3833 dòng, có {len(train_df)}"
    )
    assert len(test_df) == 548, (
        f"Test phải có 548 dòng, có {len(test_df)}"
    )

    # Bước 5 — Validate không còn leakage target
    assert "revenue" not in test_df.columns, "LEAKAGE: revenue còn trong test"
    assert "cogs"    not in test_df.columns, "LEAKAGE: cogs còn trong test"
    assert "margin"  not in test_df.columns, "LEAKAGE: margin còn trong test"

    # Bước 6 — Validate không còn observed cols trong cả hai tập
    leaked_train = [c for c in OBSERVED_COLS if c in train_df.columns]
    leaked_test  = [c for c in OBSERVED_COLS if c in test_df.columns]
    assert not leaked_train, f"LEAKAGE: observed cols còn trong train: {leaked_train}"
    assert not leaked_test,  f"LEAKAGE: observed cols còn trong test:  {leaked_test}"

    # Bước 7 — Validate Nhóm A không có NaN trong test
    group_a_cols = [
        c for c in test_df.columns
        if any(c.startswith(p) for p in GROUP_A_PREFIXES) or c in (
            "n_active_promos", "max_discount_pct", "has_stackable_promo", "pis_score"
        )
    ]
    n_nan_a = int(test_df[group_a_cols].isna().sum().sum())
    assert n_nan_a == 0, (
        f"Nhóm A feature bị NaN trong tập Test ({n_nan_a} giá trị) — "
        f"kiểm tra build_calendar_features / build_vn_holidays / "
        f"build_fourier_seasonality / build_promotion_intensity"
    )

    logger.info(
        "Validate PASS — train: %d × %d | test: %d × %d",
        len(train_df), train_df.shape[1],
        len(test_df),  test_df.shape[1],
    )

    # Bước 8 — Ghi file
    processed_dir.mkdir(parents=True, exist_ok=True)
    train_path = processed_dir / "train_features.parquet"
    test_path  = processed_dir / "test_features.parquet"

    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)

    try: 
        mimi = train_df.columns
        
        gluglu = pd.DataFrame(mimi) 
        gluglu.to_csv(processed_dir / "gluglu.csv") 
    except Exception as e: 
        print(f"{type(mimi)} ")
    logger.info(
        "Đã ghi train_features: %d dòng × %d cột → %s",
        len(train_df), train_df.shape[1], train_path,
    )
    logger.info( 
        "Đã ghi test_features:  %d dòng × %d cột → %s",
        len(test_df), test_df.shape[1], test_path,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Điểm vào chính của feature_eng.py — tạo train/test feature parquet."""
    args = parse_args()
    cfg  = load_config(args.config)
    setup_logging(cfg)

    raw_dir       = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])

    logger.info("=" * 60)
    logger.info("BẮT ĐẦU FEATURE ENGINEERING")
    logger.info("=" * 60)

    # --- Load ---
    df            = load_base_table(processed_dir)
    # FIX 1: promotions_df đã được mở rộng chu kỳ sang 2023–2024
    promotions_df = load_promotions(raw_dir)

    # -----------------------------------------------------------------------
    # NHÓM A — tính trên 4,381 ngày, KHÔNG shift, không NaN trong Test
    # -----------------------------------------------------------------------
    logger.info("--- NHÓM A: Date-based Features (không shift, không NaN) ---")
    df = build_calendar_features(df, cfg)
    df = build_vn_holidays(df, cfg)
    df = build_fourier_seasonality(df, cfg)

    # -----------------------------------------------------------------------
    # NHÓM B — VÔ HIỆU HÓA (FIX 2: không tương thích Horizon 1.5 năm)
    # Các hàm vẫn được gọi nhưng chỉ log cảnh báo và return df nguyên vẹn.
    # -----------------------------------------------------------------------
    logger.info(
        "--- NHÓM B: DISABLED (Horizon 1.5 năm — leakage + NaN toàn bộ Test) ---"
    )
    df = build_lag_features(df, cfg)
    df = build_macd_features(df, cfg)
    df = build_price_discount_elasticity(df, cfg)
    df = build_web_traffic_features(df, cfg)
    df = build_inventory_features(df, cfg)

    # -----------------------------------------------------------------------
    # PROMOTION INTENSITY — Nhóm A* (dùng promotions đã mở rộng chu kỳ)
    # -----------------------------------------------------------------------
    logger.info("--- PROMOTION INTENSITY (Nhóm A* — chu kỳ 2 năm) ---")
    df = build_promotion_intensity(df, promotions_df, cfg)

    # Downcast toàn bộ trừ target + date trước khi split
    df = downcast_df(df, exclude_cols=TARGET_COLS + ["date"])

    mem_mb = df.memory_usage(deep=True).sum() / 1e6
    logger.info(
        "Full feature DataFrame (trước split): %d dòng × %d cột | %.1f MB",
        df.shape[0], df.shape[1], mem_mb,
    )

    # -----------------------------------------------------------------------
    # SPLIT & SAVE — FIX 2: drop OBSERVED_COLS trước khi split
    # -----------------------------------------------------------------------
    logger.info("--- SPLIT & SAVE ---")
    split_and_save(df, cfg, processed_dir)

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING HOÀN TẤT")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()