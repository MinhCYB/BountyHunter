"""
data_prep.py — Agent 1: Tạo data/processed/base_table.parquet.

Pipeline:
  1. Khởi tạo skeleton date_range 2012-07-04 → 2024-07-01 (4,380 dòng).
  2. Load & chuẩn hóa toàn bộ CSV thô.
  3. Tạo daily aggregates từ dữ liệu 2012–2022.
  4. LEFT JOIN tất cả aggregates vào skeleton.
  5. Validate và ghi parquet.

Lưu ý quan trọng: 548 dòng cuối (2023–2024) sẽ NaN ở toàn bộ cột quan sát.
Đây là ĐÚNG — KHÔNG dropna, KHÔNG fillna, KHÔNG filter bỏ.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Data Prep — tạo base_table.parquet")
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
        Đường dẫn tới file config.yaml.

    Returns
    -------
    dict
        Dictionary cấu hình.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def downcast_df(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    """
    Downcast dtype để tiết kiệm RAM. Các cột trong exclude_cols giữ nguyên.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cần downcast.
    exclude_cols : list[str]
        Danh sách cột KHÔNG được downcast (target columns).

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
# Load & Normalize
# ---------------------------------------------------------------------------

def load_and_normalize(raw_dir: Path, cfg: dict) -> dict[str, pd.DataFrame]:
    """
    Load toàn bộ file CSV thô, chuẩn hóa column names, parse dates, drop constants.

    Parameters
    ----------
    raw_dir : Path
        Thư mục chứa file CSV thô.
    cfg : dict
        Config dict đọc từ config.yaml.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary {tên_bảng: DataFrame đã chuẩn hóa}.
    """
    TARGET_COLS: list[str] = [
        cfg["data"]["target_revenue"],
        cfg["data"]["target_cogs"],
        cfg["data"]["target_margin"],
    ]
    DROP_CONSTANT: list[str] = cfg["data"]["drop_constant_cols"]

    tables: dict[str, pd.DataFrame] = {}

    # sales.csv
    df_sales = pd.read_csv(raw_dir / "sales.csv", parse_dates=["Date"])
    df_sales = df_sales.rename(columns={"Date": "date", "Revenue": "revenue", "COGS": "cogs"})
    df_sales["margin"] = df_sales["cogs"] / df_sales["revenue"]  # KHÔNG downcast
    df_sales = df_sales.sort_values("date").reset_index(drop=True)
    tables["sales"] = df_sales
    logger.info("sales.csv: %d dòng, phạm vi %s → %s", len(df_sales), df_sales["date"].min().date(), df_sales["date"].max().date())

    # orders.csv
    df_orders = pd.read_csv(raw_dir / "orders.csv", parse_dates=["order_date"])
    df_orders.columns = df_orders.columns.str.lower().str.replace(" ", "_")
    for cat_col in ["order_status", "payment_method", "device_type", "order_source"]:
        if cat_col in df_orders.columns:
            df_orders[cat_col] = df_orders[cat_col].astype("category")
    df_orders = downcast_df(df_orders, exclude_cols=TARGET_COLS)
    tables["orders"] = df_orders
    logger.info("orders.csv: %d dòng", len(df_orders))

    # order_items.csv
    df_items = pd.read_csv(raw_dir / "order_items.csv")
    df_items.columns = df_items.columns.str.lower().str.replace(" ", "_")
    for cat_col in ["promo_id", "promo_id_2"]:
        if cat_col in df_items.columns:
            df_items[cat_col] = df_items[cat_col].astype("category")
    df_items = downcast_df(df_items, exclude_cols=TARGET_COLS)
    tables["order_items"] = df_items
    logger.info("order_items.csv: %d dòng", len(df_items))

    # products.csv
    df_products = pd.read_csv(raw_dir / "products.csv")
    df_products.columns = df_products.columns.str.lower().str.replace(" ", "_")
    for cat_col in ["category", "segment", "size", "color"]:
        if cat_col in df_products.columns:
            df_products[cat_col] = df_products[cat_col].astype("category")
    # products.cogs & products.price là product-level, không phải target — downcast OK
    df_products = downcast_df(df_products, exclude_cols=TARGET_COLS)
    tables["products"] = df_products
    logger.info("products.csv: %d dòng", len(df_products))

    # returns.csv
    df_returns = pd.read_csv(raw_dir / "returns.csv", parse_dates=["return_date"])
    df_returns.columns = df_returns.columns.str.lower().str.replace(" ", "_")
    df_returns = downcast_df(df_returns, exclude_cols=TARGET_COLS)
    tables["returns"] = df_returns
    logger.info("returns.csv: %d dòng", len(df_returns))

    # reviews.csv
    df_reviews = pd.read_csv(raw_dir / "reviews.csv", parse_dates=["review_date"])
    df_reviews.columns = df_reviews.columns.str.lower().str.replace(" ", "_")
    df_reviews = downcast_df(df_reviews, exclude_cols=TARGET_COLS)
    tables["reviews"] = df_reviews
    logger.info("reviews.csv: %d dòng", len(df_reviews))

    # shipments.csv
    df_shipments = pd.read_csv(raw_dir / "shipments.csv", parse_dates=["ship_date", "delivery_date"])
    df_shipments.columns = df_shipments.columns.str.lower().str.replace(" ", "_")
    df_shipments = downcast_df(df_shipments, exclude_cols=TARGET_COLS)
    tables["shipments"] = df_shipments
    logger.info("shipments.csv: %d dòng", len(df_shipments))

    # web_traffic.csv
    df_traffic = pd.read_csv(raw_dir / "web_traffic.csv", parse_dates=["date"])
    df_traffic.columns = df_traffic.columns.str.lower().str.replace(" ", "_")
    for cat_col in ["traffic_source"]:
        if cat_col in df_traffic.columns:
            df_traffic[cat_col] = df_traffic[cat_col].astype("category")
    df_traffic = downcast_df(df_traffic, exclude_cols=TARGET_COLS)
    tables["web_traffic"] = df_traffic
    logger.info("web_traffic.csv: %d dòng", len(df_traffic))

    # inventory.csv
    df_inventory = pd.read_csv(raw_dir / "inventory.csv", parse_dates=["snapshot_date"])
    df_inventory.columns = df_inventory.columns.str.lower().str.replace(" ", "_")
    # Drop constant column
    df_inventory = df_inventory.drop(columns=[c for c in DROP_CONSTANT if c in df_inventory.columns])
    for cat_col in ["category", "segment"]:
        if cat_col in df_inventory.columns:
            df_inventory[cat_col] = df_inventory[cat_col].astype("category")
    df_inventory = downcast_df(df_inventory, exclude_cols=TARGET_COLS)
    tables["inventory"] = df_inventory
    logger.info("inventory.csv: %d dòng (sau drop constant cols)", len(df_inventory))

    # customers.csv (dùng hạn chế)
    df_customers = pd.read_csv(raw_dir / "customers.csv", parse_dates=["signup_date"])
    df_customers.columns = df_customers.columns.str.lower().str.replace(" ", "_")
    df_customers = downcast_df(df_customers, exclude_cols=TARGET_COLS)
    tables["customers"] = df_customers

    return tables


# ---------------------------------------------------------------------------
# Daily Aggregates
# ---------------------------------------------------------------------------

def build_daily_aggregates(tables: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """
    Tạo tất cả daily aggregate DataFrames từ các bảng thô.
    Tất cả aggregate chỉ tính trên dữ liệu 2012–2022.

    Parameters
    ----------
    tables : dict[str, pd.DataFrame]
        Dictionary bảng đã được load_and_normalize.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary {tên_aggregate: DataFrame với cột 'date' + các cột aggregate}.
    """
    aggregates: dict[str, pd.DataFrame] = {}

    # -------------------------------------------------------------------
    # 1. sales_daily: revenue, cogs, margin từ sales.csv
    # -------------------------------------------------------------------
    df_sales = tables["sales"].copy()
    # Chỉ dữ liệu train (2012–2022)
    df_sales_train = df_sales[df_sales["date"] <= "2022-12-31"].copy()
    sales_daily = df_sales_train[["date", "revenue", "cogs", "margin"]].copy()
    aggregates["sales_daily"] = sales_daily
    logger.info("sales_daily: %d dòng", len(sales_daily))

    # -------------------------------------------------------------------
    # 2. orders_daily: từ orders + order_items + products
    # -------------------------------------------------------------------
    df_orders = tables["orders"].copy()
    df_items = tables["order_items"].copy()
    df_products = tables["products"].copy()

    # Chỉ lấy đơn hàng trong 2012–2022
    df_orders_train = df_orders[df_orders["order_date"] <= "2022-12-31"].copy()

    # Join items → products để lấy thêm info (dùng cho gross_revenue)
    df_items_with_products = df_items.merge(
        df_products[["product_id"]],
        on="product_id",
        how="left",
    )

    # Tính revenue_net per item
    df_items_with_products["revenue_net"] = (
        df_items_with_products["unit_price"] * df_items_with_products["quantity"]
        - df_items_with_products["discount_amount"]
    )

    # Aggregate order_items theo order_id
    items_agg = (
        df_items_with_products
        .groupby("order_id", observed=True)
        .agg(
            n_items=("quantity", "sum"),
            gross_revenue=("revenue_net", "sum"),
            total_discount=("discount_amount", "sum"),
            n_unique_products=("product_id", "nunique"),
        )
        .reset_index()
    )

    # Join về orders
    df_orders_rich = df_orders_train.merge(items_agg, on="order_id", how="left")

    # Flag cancelled
    df_orders_rich["is_cancelled"] = (df_orders_rich["order_status"] == "cancelled").astype("int8")

    # Aggregate theo ngày
    orders_daily = (
        df_orders_rich
        .groupby("order_date", observed=True)
        .agg(
            n_orders=("order_id", "count"),
            n_items_sold=("n_items", "sum"),
            gross_revenue=("gross_revenue", "sum"),
            total_discount=("total_discount", "sum"),
            n_unique_customers=("customer_id", "nunique"),
            n_unique_products=("n_unique_products", "sum"),
            n_cancelled=("is_cancelled", "sum"),
        )
        .reset_index()
        .rename(columns={"order_date": "date"})
    )

    # Tính derived columns — vectorized
    orders_daily["cancelled_rate"] = orders_daily["n_cancelled"] / orders_daily["n_orders"]
    orders_daily["avg_order_value"] = orders_daily["gross_revenue"] / orders_daily["n_orders"]
    orders_daily = orders_daily.drop(columns=["n_cancelled"])

    aggregates["orders_daily"] = orders_daily
    logger.info("orders_daily: %d dòng", len(orders_daily))

    # -------------------------------------------------------------------
    # 3. returns_daily
    # -------------------------------------------------------------------
    df_returns = tables["returns"].copy()
    df_returns_train = df_returns[df_returns["return_date"] <= "2022-12-31"].copy()

    returns_daily = (
        df_returns_train
        .groupby("return_date")
        .agg(
            n_returns=("return_quantity", "sum"),
            total_refund=("refund_amount", "sum"),
        )
        .reset_index()
        .rename(columns={"return_date": "date"})
    )
    aggregates["returns_daily"] = returns_daily
    logger.info("returns_daily: %d dòng", len(returns_daily))

    # -------------------------------------------------------------------
    # 4. reviews_daily
    # -------------------------------------------------------------------
    df_reviews = tables["reviews"].copy()
    df_reviews_train = df_reviews[df_reviews["review_date"] <= "2022-12-31"].copy()

    reviews_daily = (
        df_reviews_train
        .groupby("review_date")
        .agg(
            avg_rating=("rating", "mean"),
            n_reviews=("rating", "count"),
        )
        .reset_index()
        .rename(columns={"review_date": "date"})
    )
    aggregates["reviews_daily"] = reviews_daily
    logger.info("reviews_daily: %d dòng", len(reviews_daily))

    # -------------------------------------------------------------------
    # 5. shipment_daily
    # -------------------------------------------------------------------
    df_shipments = tables["shipments"].copy()
    df_shipments_train = df_shipments[df_shipments["ship_date"] <= "2022-12-31"].copy()

    # Tính delivery_days — vectorized
    df_shipments_train["delivery_days"] = (
        df_shipments_train["delivery_date"] - df_shipments_train["ship_date"]
    ).dt.days

    shipment_daily = (
        df_shipments_train
        .groupby("ship_date")
        .agg(avg_delivery_days=("delivery_days", "mean"))
        .reset_index()
        .rename(columns={"ship_date": "date"})
    )
    aggregates["shipment_daily"] = shipment_daily
    logger.info("shipment_daily: %d dòng", len(shipment_daily))

    # -------------------------------------------------------------------
    # 6. traffic_daily: giữ nguyên daily, aggregate nếu có nhiều dòng/ngày
    # -------------------------------------------------------------------
    df_traffic = tables["web_traffic"].copy()
    df_traffic_train = df_traffic[df_traffic["date"] <= "2022-12-31"].copy()

    traffic_daily = (
        df_traffic_train
        .groupby("date")
        .agg(
            sessions=("sessions", "sum"),
            unique_visitors=("unique_visitors", "sum"),
            page_views=("page_views", "sum"),
            bounce_rate=("bounce_rate", "mean"),
            avg_session_duration_sec=("avg_session_duration_sec", "mean"),
        )
        .reset_index()
    )
    aggregates["traffic_daily"] = traffic_daily
    logger.info("traffic_daily: %d dòng", len(traffic_daily))

    # -------------------------------------------------------------------
    # 7. inventory_daily: forward-fill monthly snapshot → daily (chỉ 2012–2022)
    # -------------------------------------------------------------------
    df_inventory = tables["inventory"].copy()
    df_inventory_train = df_inventory[df_inventory["snapshot_date"] <= "2022-12-31"].copy()

    # Aggregate theo snapshot_date (tất cả category)
    inv_agg = (
        df_inventory_train
        .groupby("snapshot_date", observed=True)
        .agg(
            avg_fill_rate=("fill_rate", "mean"),
            avg_stockout_days=("stockout_days", "mean"),
            pct_stockout_skus=("stockout_flag", "mean"),
            avg_sell_through=("sell_through_rate", "mean"),
        )
        .reset_index()
        .rename(columns={"snapshot_date": "date"})
    )

    # Forward-fill monthly snapshot thành daily — chỉ trong 2012–2022
    date_range_train = pd.date_range(start="2012-07-04", end="2022-12-31", freq="D")
    inv_skeleton = pd.DataFrame({"date": date_range_train})
    inv_daily = inv_skeleton.merge(inv_agg, on="date", how="left")
    inv_daily = inv_daily.sort_values("date").reset_index(drop=True)

    # Forward-fill (chỉ trong phạm vi train, không sang 2023+)
    fill_cols = ["avg_fill_rate", "avg_stockout_days", "pct_stockout_skus", "avg_sell_through"]
    inv_daily[fill_cols] = inv_daily[fill_cols].ffill()

    aggregates["inventory_daily"] = inv_daily
    logger.info("inventory_daily: %d dòng (sau forward-fill)", len(inv_daily))

    return aggregates


# ---------------------------------------------------------------------------
# Build Base Table
# ---------------------------------------------------------------------------

def build_base_table(
    skeleton: pd.DataFrame,
    aggregates: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    LEFT JOIN tất cả daily aggregates vào skeleton theo cột 'date'.

    Parameters
    ----------
    skeleton : pd.DataFrame
        DataFrame với cột 'date' từ 2012-07-04 đến 2024-07-01.
    aggregates : dict[str, pd.DataFrame]
        Dictionary aggregate DataFrames, mỗi cái có cột 'date'.

    Returns
    -------
    pd.DataFrame
        base_table với toàn bộ dữ liệu đã join.
        548 dòng cuối (2023–2024) sẽ NaN ở toàn bộ cột quan sát — đây là ĐÚNG.
    """
    base = skeleton.copy()

    join_order = [
        "sales_daily",
        "orders_daily",
        "returns_daily",
        "reviews_daily",
        "shipment_daily",
        "traffic_daily",
        "inventory_daily",
    ]

    for agg_name in join_order:
        if agg_name not in aggregates:
            logger.warning("Không tìm thấy aggregate '%s', bỏ qua", agg_name)
            continue
        df_agg = aggregates[agg_name]
        base = base.merge(df_agg, on="date", how="left")
        logger.info("Sau join %s: base có %d cột", agg_name, base.shape[1])

    return base.sort_values("date").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Validate
# ---------------------------------------------------------------------------

def validate_base_table(df: pd.DataFrame) -> None:
    """
    Kiểm tra tính toàn vẹn của base_table trước khi ghi ra file.

    Parameters
    ----------
    df : pd.DataFrame
        base_table đã được build_base_table tạo ra.

    Raises
    ------
    AssertionError
        Nếu bất kỳ điều kiện nào không thỏa mãn.
    """
    expected_start = pd.Timestamp("2012-07-04")
    expected_end = pd.Timestamp("2024-07-01")

    # Assert số dòng
    skeleton_len = len(pd.date_range(start=expected_start, end=expected_end, freq="D"))
    if len(df) != skeleton_len:
        logger.error("Số dòng sai: %d (kỳ vọng %d)", len(df), skeleton_len)
        raise AssertionError(f"Số dòng base_table sai: {len(df)} != {skeleton_len}")

    # Assert date range
    if df["date"].min() != expected_start:
        logger.error("date.min() sai: %s (kỳ vọng %s)", df["date"].min(), expected_start)
        raise AssertionError(f"date.min() sai: {df['date'].min()}")

    if df["date"].max() != expected_end:
        logger.error("date.max() sai: %s (kỳ vọng %s)", df["date"].max(), expected_end)
        raise AssertionError(f"date.max() sai: {df['date'].max()}")

    # Assert vùng Test (2023–2024) có revenue = NaN
    if "revenue" in df.columns:
        test_mask = df["date"] >= pd.Timestamp("2023-01-01")
        n_non_null_test = int(df.loc[test_mask, "revenue"].notna().sum())
        if n_non_null_test > 0:
            logger.error("Vùng Test có %d dòng revenue không NaN — vi phạm thiết kế!", n_non_null_test)
            raise AssertionError(f"Vùng Test phải NaN nhưng có {n_non_null_test} dòng revenue không NaN")
        else:
            logger.info("✓ Vùng Test (2023–2024): toàn bộ revenue = NaN — đúng thiết kế")

    logger.info("✓ Validate base_table PASS: %d dòng, %d cột", len(df), df.shape[1])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Điểm vào chính: load config, chạy toàn bộ pipeline data_prep,
    validate và ghi base_table.parquet.
    """
    args = parse_args()
    cfg = load_config(args.config)

    # Setup logging
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    from datetime import datetime
    log_file = log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"]),
        format=cfg["logging"]["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    raw_dir = Path(cfg["paths"]["raw_dir"])
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # BƯỚC 0: Khởi tạo skeleton — BẮT BUỘC LÀM ĐẦU TIÊN
    # -----------------------------------------------------------------------
    train_start: str = cfg["data"]["train_start"]  # "2012-07-04"
    test_end: str = cfg["data"]["test_end"]         # "2024-07-01"

    skeleton = pd.DataFrame({
        "date": pd.date_range(start=train_start, end=test_end, freq="D")
    })
    logger.info("Skeleton khởi tạo: %d dòng (%s → %s)", len(skeleton), train_start, test_end)

    # -----------------------------------------------------------------------
    # BƯỚC 1: Load & chuẩn hóa
    # -----------------------------------------------------------------------
    logger.info("=== BƯỚC 1: Load & Normalize ===")
    tables = load_and_normalize(raw_dir, cfg)

    # -----------------------------------------------------------------------
    # BƯỚC 2: Tạo daily aggregates
    # -----------------------------------------------------------------------
    logger.info("=== BƯỚC 2: Build Daily Aggregates ===")
    aggregates = build_daily_aggregates(tables)

    # -----------------------------------------------------------------------
    # BƯỚC 3: LEFT JOIN vào skeleton
    # -----------------------------------------------------------------------
    logger.info("=== BƯỚC 3: Build Base Table (LEFT JOIN) ===")
    base_table = build_base_table(skeleton, aggregates)

    # -----------------------------------------------------------------------
    # BƯỚC 4: Validate
    # -----------------------------------------------------------------------
    logger.info("=== BƯỚC 4: Validate ===")
    validate_base_table(base_table)

    # Log thống kê
    n_null_revenue = int(base_table["revenue"].isna().sum()) if "revenue" in base_table.columns else -1
    mem_mb = base_table.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(
        "base_table: %d dòng | %d cột | %.1f MB memory | %d dòng NaN ở revenue",
        len(base_table), base_table.shape[1], mem_mb, n_null_revenue,
    )

    # -----------------------------------------------------------------------
    # Ghi output
    # -----------------------------------------------------------------------
    output_path = processed_dir / "base_table.parquet"
    base_table.to_parquet(output_path, index=False)

    logger.info("Đã ghi base_table ra %s", output_path)
    logger.info("Data Prep hoàn tất")


if __name__ == "__main__":
    main()