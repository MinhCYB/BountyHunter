"""
data_audit.py — Agent 1: Kiểm tra chất lượng dữ liệu thô.

Chạy độc lập. Không ghi file output ngoài log và outputs/qa_report.csv.
Phạm vi kiểm tra: chỉ dữ liệu 2012–2022. Không báo lỗi thiếu data 2023–2024.
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
    parser = argparse.ArgumentParser(description="Data Audit — kiểm tra chất lượng dữ liệu thô")
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


def _make_record(
    table: str,
    column: str,
    check_name: str,
    status: str,
    n_violations: int,
    detail: str,
) -> dict:
    """
    Tạo một record kết quả kiểm tra.

    Parameters
    ----------
    table : str
        Tên bảng đang kiểm tra.
    column : str
        Tên cột đang kiểm tra (hoặc chuỗi rỗng nếu check toàn bảng).
    check_name : str
        Tên loại check.
    status : str
        Kết quả: PASS | WARNING | ERROR.
    n_violations : int
        Số dòng vi phạm.
    detail : str
        Mô tả ngắn hoặc ví dụ vi phạm.

    Returns
    -------
    dict
        Record kết quả theo schema qa_report.
    """
    return {
        "table": table,
        "column": column,
        "check_name": check_name,
        "status": status,
        "n_violations": n_violations,
        "detail": detail,
    }


# ---------------------------------------------------------------------------
# Các hàm kiểm tra chính
# ---------------------------------------------------------------------------

def audit_table(
    df: pd.DataFrame,
    table_name: str,
    expected_rows: int,
    nullable_cols: list[str],
) -> list[dict]:
    """
    Kiểm tra row count và null rate của một bảng.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame cần kiểm tra.
    table_name : str
        Tên bảng (dùng trong log và report).
    expected_rows : int
        Số dòng kỳ vọng theo Data Contract. -1 = bỏ qua check này.
    nullable_cols : list[str]
        Danh sách cột NULLABLE_BY_DESIGN — bỏ qua khi check null.

    Returns
    -------
    list[dict]
        Danh sách record kết quả kiểm tra.
    """
    records: list[dict] = []

    # --- Row count ---
    actual_rows = len(df)
    if expected_rows > 0:
        if actual_rows != expected_rows:
            msg = f"Kỳ vọng {expected_rows} dòng, thực tế {actual_rows} dòng"
            logger.warning("[%s] Row count lệch: %s", table_name, msg)
            records.append(_make_record(table_name, "", "row_count", "WARNING", abs(actual_rows - expected_rows), msg))
        else:
            records.append(_make_record(table_name, "", "row_count", "PASS", 0, f"{actual_rows} dòng đúng kỳ vọng"))

    # --- Null rate mỗi cột ---
    for col in df.columns:
        if col in nullable_cols:
            continue
        n_null = int(df[col].isna().sum())
        null_rate = n_null / len(df) if len(df) > 0 else 0.0
        if n_null > 0:
            msg = f"null_rate={null_rate:.2%} ({n_null}/{len(df)})"
            logger.warning("[%s.%s] Có null không mong muốn: %s", table_name, col, msg)
            records.append(_make_record(table_name, col, "null_check", "WARNING", n_null, msg))
        else:
            records.append(_make_record(table_name, col, "null_check", "PASS", 0, "Không có null"))

    return records


def audit_referential_integrity(
    df_child: pd.DataFrame,
    df_parent: pd.DataFrame,
    fk_col: str,
    pk_col: str,
    table_name: str,
) -> list[dict]:
    """
    Kiểm tra FK integrity: mọi giá trị fk_col trong df_child phải tồn tại trong pk_col của df_parent.

    Parameters
    ----------
    df_child : pd.DataFrame
        Bảng con chứa foreign key.
    df_parent : pd.DataFrame
        Bảng cha chứa primary key.
    fk_col : str
        Tên cột FK trong df_child.
    pk_col : str
        Tên cột PK trong df_parent.
    table_name : str
        Tên để hiển thị trong report (ví dụ: "orders→customers").

    Returns
    -------
    list[dict]
        Danh sách record kết quả.
    """
    records: list[dict] = []

    valid_pks = set(df_parent[pk_col].dropna().unique())
    fk_series = df_child[fk_col].dropna()
    orphan_mask = ~fk_series.isin(valid_pks)
    n_orphan = int(orphan_mask.sum())

    check_name = f"fk_integrity_{fk_col}"
    if n_orphan > 0:
        examples = fk_series[orphan_mask].unique()[:5].tolist()
        msg = f"{n_orphan} giá trị orphan, ví dụ: {examples}"
        logger.warning("[%s] FK integrity lỗi: %s", table_name, msg)
        records.append(_make_record(table_name, fk_col, check_name, "ERROR", n_orphan, msg))
    else:
        records.append(_make_record(table_name, fk_col, check_name, "PASS", 0, "Toàn bộ FK hợp lệ"))

    return records


def audit_sales_constraints(df_sales: pd.DataFrame) -> list[dict]:
    """
    Kiểm tra các ràng buộc đặc thù của sales.csv:
      - revenue > 0 và cogs > 0.
      - Date range nằm trong 04/07/2012 – 31/12/2022.

    Parameters
    ----------
    df_sales : pd.DataFrame
        DataFrame sales đã được chuẩn hóa (cột: date, revenue, cogs).

    Returns
    -------
    list[dict]
        Danh sách record kết quả.
    """
    records: list[dict] = []
    table = "sales"

    # --- revenue > 0 ---
    n_bad_rev = int((df_sales["revenue"] <= 0).sum())
    if n_bad_rev > 0:
        msg = f"{n_bad_rev} dòng có revenue <= 0"
        logger.error("[sales] %s", msg)
        records.append(_make_record(table, "revenue", "positive_check", "ERROR", n_bad_rev, msg))
    else:
        records.append(_make_record(table, "revenue", "positive_check", "PASS", 0, "revenue > 0 toàn bộ"))

    # --- cogs > 0 ---
    n_bad_cogs = int((df_sales["cogs"] <= 0).sum())
    if n_bad_cogs > 0:
        msg = f"{n_bad_cogs} dòng có cogs <= 0"
        logger.error("[sales] %s", msg)
        records.append(_make_record(table, "cogs", "positive_check", "ERROR", n_bad_cogs, msg))
    else:
        records.append(_make_record(table, "cogs", "positive_check", "PASS", 0, "cogs > 0 toàn bộ"))

    # --- Date range ---
    min_allowed = pd.Timestamp("2012-07-04")
    max_allowed = pd.Timestamp("2022-12-31")
    out_of_range = df_sales[(df_sales["date"] < min_allowed) | (df_sales["date"] > max_allowed)]
    n_out = len(out_of_range)
    if n_out > 0:
        examples = out_of_range["date"].dt.date.unique()[:5].tolist()
        msg = f"{n_out} dòng ngoài khoảng 2012-07-04 – 2022-12-31, ví dụ: {examples}"
        logger.error("[sales] Date range lỗi: %s", msg)
        records.append(_make_record(table, "date", "date_range", "ERROR", n_out, msg))
    else:
        records.append(_make_record(table, "date", "date_range", "PASS", 0, "Date range trong giới hạn"))

    return records


def audit_products_constraints(df_products: pd.DataFrame) -> list[dict]:
    """
    Kiểm tra ràng buộc products.cogs < products.price.

    Parameters
    ----------
    df_products : pd.DataFrame
        DataFrame products đã được chuẩn hóa.

    Returns
    -------
    list[dict]
        Danh sách record kết quả.
    """
    records: list[dict] = []
    table = "products"

    violate_mask = df_products["cogs"] >= df_products["price"]
    n_violate = int(violate_mask.sum())
    if n_violate > 0:
        examples = df_products.loc[violate_mask, "product_id"].tolist()[:5]
        msg = f"{n_violate} sản phẩm có cogs >= price, ví dụ product_id: {examples}"
        logger.error("[products] Ràng buộc cogs < price bị vi phạm: %s", msg)
        records.append(_make_record(table, "cogs/price", "cogs_lt_price", "ERROR", n_violate, msg))
    else:
        records.append(_make_record(table, "cogs/price", "cogs_lt_price", "PASS", 0, "cogs < price toàn bộ"))

    return records


def audit_inventory_constants(df_inventory: pd.DataFrame) -> list[dict]:
    """
    Kiểm tra inventory.reorder_flag luôn = 0 (CONSTANT).

    Parameters
    ----------
    df_inventory : pd.DataFrame
        DataFrame inventory đã được chuẩn hóa.

    Returns
    -------
    list[dict]
        Danh sách record kết quả.
    """
    records: list[dict] = []
    table = "inventory"

    if "reorder_flag" not in df_inventory.columns:
        logger.info("[inventory] reorder_flag không tồn tại trong DataFrame (đã drop đúng cách)")
        records.append(_make_record(table, "reorder_flag", "constant_check", "PASS", 0, "Cột đã được drop theo thiết kế"))
        return records

    unique_vals = df_inventory["reorder_flag"].unique()
    if len(unique_vals) == 1 and unique_vals[0] == 0:
        logger.info("[inventory] reorder_flag xác nhận CONSTANT = 0, có thể drop an toàn")
        records.append(_make_record(table, "reorder_flag", "constant_check", "PASS", 0, "CONSTANT = 0, an toàn để drop"))
    else:
        msg = f"Giá trị unique: {unique_vals.tolist()}"
        logger.warning("[inventory] reorder_flag KHÔNG constant: %s", msg)
        records.append(_make_record(table, "reorder_flag", "constant_check", "WARNING", len(unique_vals) - 1, msg))

    return records


def audit_promo_id_2(df_order_items: pd.DataFrame) -> list[dict]:
    """
    Kiểm tra order_items.promo_id_2 có null rate ≈ 100%.

    Parameters
    ----------
    df_order_items : pd.DataFrame
        DataFrame order_items đã được chuẩn hóa.

    Returns
    -------
    list[dict]
        Danh sách record kết quả.
    """
    records: list[dict] = []
    table = "order_items"

    if "promo_id_2" not in df_order_items.columns:
        records.append(_make_record(table, "promo_id_2", "high_null_check", "WARNING", 0, "Cột không tồn tại"))
        return records

    null_rate = df_order_items["promo_id_2"].isna().mean()
    if null_rate < 0.95:
        msg = f"null_rate={null_rate:.2%} < 95% — không còn HIGH NULL như kỳ vọng"
        logger.warning("[order_items.promo_id_2] %s", msg)
        records.append(_make_record(table, "promo_id_2", "high_null_check", "WARNING", 0, msg))
    else:
        records.append(_make_record(table, "promo_id_2", "high_null_check", "PASS", 0, f"null_rate={null_rate:.2%} ≥ 95% đúng kỳ vọng"))

    return records


# ---------------------------------------------------------------------------
# Save output
# ---------------------------------------------------------------------------

def save_qa_report(records: list[dict], output_dir: Path) -> None:
    """
    Ghi danh sách record kiểm tra ra file qa_report.csv.

    Parameters
    ----------
    records : list[dict]
        Danh sách kết quả từ các hàm audit_*.
    output_dir : Path
        Thư mục output (sẽ tạo nếu chưa có).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "qa_report.csv"

    df_report = pd.DataFrame(records, columns=["table", "column", "check_name", "status", "n_violations", "detail"])
    df_report.to_csv(report_path, index=False, encoding="utf-8")

    n_error = int((df_report["status"] == "ERROR").sum())
    n_warning = int((df_report["status"] == "WARNING").sum())
    n_pass = int((df_report["status"] == "PASS").sum())
    logger.info(
        "QA Report đã ghi ra %s — PASS: %d | WARNING: %d | ERROR: %d",
        report_path, n_pass, n_warning, n_error,
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Điểm vào chính: load config, load data, chạy toàn bộ audit, ghi report."""
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
    output_dir = Path(cfg["paths"]["output_dir"])
    nullable_cols: list[str] = cfg["data"]["nullable_by_design"]

    all_records: list[dict] = []

    # --- Load các bảng ---
    logger.info("Bắt đầu load dữ liệu thô từ %s", raw_dir)

    df_sales = pd.read_csv(raw_dir / "sales.csv", parse_dates=["Date"])
    df_sales = df_sales.rename(columns={"Date": "date", "Revenue": "revenue", "COGS": "cogs"})

    df_orders = pd.read_csv(raw_dir / "orders.csv", parse_dates=["order_date"])
    df_orders.columns = df_orders.columns.str.lower().str.replace(" ", "_")

    df_order_items = pd.read_csv(raw_dir / "order_items.csv")
    df_order_items.columns = df_order_items.columns.str.lower().str.replace(" ", "_")

    df_products = pd.read_csv(raw_dir / "products.csv")
    df_products.columns = df_products.columns.str.lower().str.replace(" ", "_")

    df_customers = pd.read_csv(raw_dir / "customers.csv", parse_dates=["signup_date"])
    df_customers.columns = df_customers.columns.str.lower().str.replace(" ", "_")

    df_inventory = pd.read_csv(raw_dir / "inventory.csv", parse_dates=["snapshot_date"])
    df_inventory.columns = df_inventory.columns.str.lower().str.replace(" ", "_")

    logger.info("Load xong tất cả bảng cần audit")

    # --- Audit 1: Row count & null rate ---
    # Expected rows theo Data Contract (PART_I mục 3)
    all_records += audit_table(df_sales, "sales", expected_rows=3833, nullable_cols=nullable_cols)
    all_records += audit_table(df_orders, "orders", expected_rows=646945, nullable_cols=nullable_cols)
    all_records += audit_table(df_order_items, "order_items", expected_rows=-1, nullable_cols=nullable_cols)
    all_records += audit_table(df_products, "products", expected_rows=2412, nullable_cols=nullable_cols)
    all_records += audit_table(df_inventory, "inventory", expected_rows=60247, nullable_cols=nullable_cols)

    # --- Audit 2: Sales constraints ---
    all_records += audit_sales_constraints(df_sales)

    # --- Audit 3: Products constraints ---
    all_records += audit_products_constraints(df_products)

    # --- Audit 4: inventory.reorder_flag constant ---
    all_records += audit_inventory_constants(df_inventory)

    # --- Audit 5: promo_id_2 high null ---
    all_records += audit_promo_id_2(df_order_items)

    # --- Audit 6: FK integrity orders → customers ---
    all_records += audit_referential_integrity(
        df_child=df_orders,
        df_parent=df_customers,
        fk_col="customer_id",
        pk_col="customer_id",
        table_name="orders→customers",
    )

    # --- Audit 7: FK integrity order_items → orders ---
    all_records += audit_referential_integrity(
        df_child=df_order_items,
        df_parent=df_orders,
        fk_col="order_id",
        pk_col="order_id",
        table_name="order_items→orders",
    )

    # --- Ghi report ---
    save_qa_report(all_records, output_dir)
    logger.info("Data Audit hoàn tất")


if __name__ == "__main__":
    main()