"""
================================================================================
MODULE: DATA AUDIT & QUALITY ASSURANCE (QA)
================================================================================

Mục đích: 
Thực hiện kiểm toán và đánh giá toàn diện chất lượng dữ liệu thô (Raw Data) trước khi 
chuyển sang phân hệ tiền xử lý (Data Preprocessing). Module có nhiệm vụ phát hiện, 
phân loại và ghi nhận các điểm dị thường, vi phạm logic kinh doanh, và sai lệch luồng tiền.

Phạm vi Kiểm định Cốt lõi (Core Audit Scope):

1. Kiểm định Cấu trúc (Structural Validation):
   - Nhận diện và thống kê dữ liệu khuyết thiếu (Missing/Null values) ngoài thiết kế.
   - Phát hiện bản ghi trùng lặp (Duplicates) và sai lệch định dạng (Datetime format).

2. Kiểm định Logic Nghiệp vụ (Business Rule Verification):
   - Ràng buộc vận hành: Giám sát các chỉ số phi lý (Số lượng <= 0, Tiền hoàn < 0).
   - Biên lợi nhuận: Cảnh báo bất thường khi Giá vốn (COGS) vượt Giá bán (Price).
   - Logic Khuyến mãi (Promo Logic Verification):
     + [MOV]: Đảm bảo tổng giá trị đơn hàng đạt ngưỡng tối thiểu (min_order_value).
     + [Stackable Rule]: Phát hiện gian lận khi áp mã phụ lúc mã chính cấm cộng dồn.
     + [Combo Math]: Đối soát tính chính xác của cột discount_amount dựa trên công 
       thức toán học linh hoạt (Fixed/Percentage) cho cả đơn và đa mã khuyến mãi.

3. Toàn vẹn Tham chiếu (Referential Integrity):
   - Truy vết các bản ghi mồ côi (Orphaned records) bị đứt gãy quan hệ 
     (Ví dụ: Giao dịch phát sinh với sản phẩm không tồn tại trong danh mục).

4. Đối soát Tài chính Macro (Financial Reconciliation):
   - Đối chiếu luồng Doanh thu Gộp (Gross Revenue = quantity * unit_price) giữa 
     dữ liệu giao dịch chi tiết (order_items) và báo cáo hệ thống (sales).
   - Đối chiếu luồng Giá vốn hàng bán (COGS) giữa các phân hệ.

Đầu ra (Output): 
Xuất báo cáo kiểm toán `qa_report.csv` cung cấp tọa độ lỗi chính xác (Order IDs, 
Dates, Indices) kèm đề xuất hướng khắc phục minh bạch cho luồng Data Preprocessing.
"""

import pandas as pd 
import numpy as np
from src.config import config 

# ==========================================
# CÁC HÀM TIỆN ÍCH (HELPER)
# ==========================================
logger = config.get_logger(__name__)

def log_issue(report_list: list, category: str, table: str, columns: str, issue: str, 
              impact_count: int, sample_evidence: str = "", note: str = ""): 
    """
    Ghi nhận một phát hiện lỗi vào danh sách báo cáo kiểm toán và xuất log tức thời.

    Hàm này đóng vai trò là điểm ghi nhận tập trung duy nhất (Single Point of Truth)
    cho toàn bộ module audit. Mọi hàm kiểm tra (_check_*) đều phải dùng hàm này
    thay vì tự append vào list để đảm bảo schema báo cáo nhất quán.

    Args:
        report_list (list): Danh sách tích lũy các issue trong phiên kiểm tra hiện tại.
                            Hàm sẽ append trực tiếp vào list này (in-place).
        category (str): Phân loại lỗi theo tầng kiểm định. Ví dụ:
                        "Structural", "Business_Rules: MOV", "Business_Rules: Math",
                        "Referential", "Reconciliation: Revenue Mismatch".
        table (str): Tên bảng (CSV stem) chứa dữ liệu bị lỗi. Ví dụ: "order_items".
        columns (str): Tên cột hoặc nhóm cột liên quan. Ví dụ: "promo_id", "cogs, price".
                       Dùng "ALL_COLUMNS" nếu lỗi ảnh hưởng toàn dòng (trùng lặp).
        issue (str): Mô tả ngắn gọn, rõ ràng về bản chất lỗi. Nên bắt đầu bằng động từ.
                     Ví dụ: "Giá vốn (COGS) >= Giá bán", "Trùng lặp dòng".
        impact_count (int): Số dòng bị ảnh hưởng. Dùng len(df) hoặc mask.sum().
        sample_evidence (str): Bằng chứng cụ thể dạng chuỗi để truy vết nhanh.
                               Khuyến nghị format: "Indices: [0, 1, 2]"
                               hoặc "Order IDs: [101, 202]" hoặc "Dates: ['2023-01-01']".
                               Mặc định: chuỗi rỗng nếu không có evidence.
        note (str): Hướng xử lý gợi ý cho luồng Data Preprocessing.
                    Ví dụ: "Drop dòng", "Tính lại theo công thức", "Cân nhắc Fill".
                    Mặc định: chuỗi rỗng.

    Returns:
        None. Kết quả được ghi trực tiếp vào report_list (side effect).

    Side Effects:
        - Append một dict vào report_list với các keys chuẩn:
          Table, Columns, Category, Issue Description, Impact Rows, Sample Evidence, Note.
        - Gọi logger.info() để xuất log tức thời ra console và file log.

    Example:
        >>> issues = []
        >>> log_issue(issues, "Structural", "orders", "order_date",
        ...           "Sai định dạng Date", 3, "Indices: [10, 45, 78]", "Kiểm tra lại")
        >>> len(issues)
        1
    """
    report_list.append({
        "Table": table, 
        "Columns": columns,
        "Category": category, 
        "Issue Description": issue, 
        "Impact Rows": impact_count, 
        "Sample Evidence": sample_evidence,
        "Note": note
    })
    logger.info(f"LỖI: |{table:>12}|{columns:>20}|{sample_evidence:>53}|{category:>20}|{issue:>60}|")

def load_data(source_tables: list) -> dict: 
    """
    Quét thư mục RAW và tải các bảng CSV được chỉ định vào bộ nhớ.

    Hàm duyệt qua tất cả file .csv trong config.RAW theo thứ tự alphabet,
    chỉ load những file có tên (stem) xuất hiện trong source_tables.
    Mỗi file được đọc bằng pd.read_csv với low_memory=False để tránh
    cảnh báo DtypeWarning trên các cột có kiểu dữ liệu hỗn hợp.

    Nếu một file không đọc được (lỗi encoding, file corrupt, v.v.),
    lỗi sẽ được log và bỏ qua — các bảng còn lại vẫn tiếp tục được load.

    Args:
        source_tables (list[str]): Danh sách tên bảng cần load (không có đuôi .csv).
                                   Thường lấy từ config.SOURCE_TABLES.

    Returns:
        dict[str, pd.DataFrame]: Dictionary ánh xạ tên bảng → DataFrame.
                                  Chỉ chứa các bảng load thành công.
                                  Bảng load thất bại sẽ vắng mặt trong dict.

    Logs:
        INFO  — Mỗi bảng được load thành công.
        ERROR — Mỗi bảng bị lỗi khi đọc, kèm thông báo exception.

    Example:
        >>> tables = load_data(["orders", "order_items", "products"])
        >>> list(tables.keys())
        ['order_items', 'orders', 'products']
    """ 
    tables = {}
    for file in sorted(config.RAW.glob("*.csv")):
        try: 
            table_name = file.stem 

            if table_name in source_tables: 
                tables[table_name] = pd.read_csv(file, low_memory=False) 
                logger.info(f"Load thành công {table_name}")
        except Exception as e: 
            logger.error(f"Lỗi tải dữ liệu từ file \"{table_name}.csv\": {e}")
    return tables 

# ==========================================
# CÁC HÀM KIỂM TRA LÕI (AUDIT CHECKS)
# ==========================================

### Kiểm tra cấu trúc ### 

def _check_duplicates(df: pd.DataFrame, table_name: str, category: str, report_list: list): 
    """
    Phát hiện các dòng trùng lặp hoàn toàn (identical across ALL columns) trong bảng.

    Sử dụng pd.DataFrame.duplicated() với keep='first' (mặc định) — dòng xuất hiện
    lần đầu được giữ lại, các dòng lặp lại sau đó bị đánh dấu là duplicate.

    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra.
        table_name (str): Tên bảng, dùng để ghi vào báo cáo.
        category (str): Nhãn phân loại lỗi, thường là "Structural".
        report_list (list): Danh sách tích lũy issue, được truyền vào log_issue.

    Logs Issue:
        Columns  : "ALL_COLUMNS"
        Note     : "Drop" — xử lý bằng cách xóa dòng trùng trong preprocessing.
    """
    dups = df[df.duplicated()] 
    dup_count = dups.shape[0] 
    if dup_count > 0: 
        samples = dups.head(3).index.tolist()
        log_issue(report_list, category, table_name, "ALL_COLUMNS", "Trùng lặp dòng", dup_count, str(samples), "Drop") 

def _check_missing(df: pd.DataFrame, table_name: str, category: str, report_list: list): 
    """
    Phát hiện các cột có giá trị null nằm ngoài danh sách NULLABLE_BY_DESIGN.

    Với mỗi cột không thuộc nullable whitelist, hàm tính tỷ lệ null và phân loại:
    - Nghiêm trọng (ratio > 30%): Gợi ý Drop cột.
    - Nhẹ (ratio <= 30%)        : Gợi ý Fill trong preprocessing.

    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra.
        table_name (str): Tên bảng, dùng để tra cứu NULLABLE_BY_DESIGN và ghi báo cáo.
        category (str): Nhãn phân loại lỗi, thường là "Structural".
        report_list (list): Danh sách tích lũy issue.

    Config dependency:
        config.NULLABLE_BY_DESIGN (dict): Map tên bảng → list cột được phép null.
        Các cột có trong list này sẽ được bỏ qua hoàn toàn dù có null.
    """
    total = len(df) 
    nullable = config.NULLABLE_BY_DESIGN.get(table_name, []) 
    for col in df.columns: 
        n_null = df[col].isna().sum()
        if n_null > 0 and col not in nullable: 
            ratio = n_null / total 
            sample_idx = df[df[col].isna()].index.tolist()[:3]
            if ratio > 0.3: 
                log_issue(report_list, category, table_name, col, f"Cột \"{col}\" thiếu data nghiêm trọng ({(ratio*100):.1f}%)", 
                            n_null, f"Indices: {sample_idx}", "Cân nhắc Drop")
            else: 
                log_issue(report_list, category, table_name, col, f"Cột \"{col}\" thiếu data nhẹ ({(ratio*100):.1f}%)", 
                            n_null, f"Indices: {sample_idx}", "Tính phương án fill trong prep")

def _check_datetime_format(df: pd.DataFrame, table_name: str, category: str, report_list: list): 
    """
    Phát hiện giá trị ngày tháng có định dạng không hợp lệ trong các cột date.

    Chỉ kiểm tra các cột có dtype == "object" (tức là chưa được parse sang datetime).
    Sử dụng pd.to_datetime(..., errors="coerce") để parse thử — giá trị không parse
    được sẽ trở thành NaT. Kết hợp với mask notna() để phân biệt:
    - NaT do parse lỗi (= bad format) → flag lỗi.
    - NaT do giá trị gốc đã là null   → không flag (đã được _check_missing xử lý).

    Args:
        df (pd.DataFrame): DataFrame cần kiểm tra.
        table_name (str): Tên bảng, dùng để tra cứu DATE_COLS và ghi báo cáo.
        category (str): Nhãn phân loại lỗi, thường là "Structural".
        report_list (list): Danh sách tích lũy issue.

    Config dependency:
        config.DATE_COLS (dict): Map tên bảng → list tên cột ngày tháng cần kiểm tra.
        Bảng không có trong config.DATE_COLS sẽ được bỏ qua.
    """
    date_cols = config.DATE_COLS.get(table_name, [])
    for col in date_cols:
        if col in df.columns and df[col].dtype == "object":
            temp_parsed = pd.to_datetime(df[col], errors="coerce")
            bad_format_mask = temp_parsed.isna() & df[col].notna()
            n_bad_date = bad_format_mask.sum()
            
            if n_bad_date > 0:
                bad_indices = df[bad_format_mask].index.tolist()[:5]
                bad_values = df.loc[bad_indices, col].tolist()
                evidence = f"Indices: {bad_indices} | Values: {bad_values}"
                log_issue(report_list, category, table_name, col, "Sai định dạng Date", 
                          n_bad_date, evidence, "Kiểm tra lại")

def check_structural(tables: dict, audit_report: list, source_tables: list, category: str = "Structural"): 
    """
    Orchestrator kiểm định cấu trúc — chạy tuần tự 3 bước: duplicate, missing, datetime.

    Với mỗi bảng trong source_tables, lần lượt gọi:
    1. _check_duplicates  — phát hiện dòng trùng lặp.
    2. _check_missing     — phát hiện null ngoài thiết kế.
    3. _check_datetime_format — phát hiện định dạng ngày sai.

    Args:
        tables (dict[str, pd.DataFrame]): Dictionary bảng đã load từ load_data().
        audit_report (list): Danh sách tổng hợp toàn bộ issue xuyên suốt pipeline.
                             Kết quả từ hàm này sẽ được extend vào list này.
        source_tables (list[str]): Danh sách bảng cần kiểm tra. Bảng có trong tables
                                   nhưng không có trong source_tables sẽ bị skip.
        category (str): Nhãn phân loại, mặc định "Structural".

    Returns:
        int: Số lượng issue phát hiện được trong tầng kiểm định này.
             Dùng để tổng hợp SUMMARY log trong run_audit().
    """
    logger.info(" 1. Kiểm định cấu trúc") 
    report_list = []

    for name, df in tables.items(): 
        if name not in source_tables: 
            continue 
        _check_duplicates(df, name, category, report_list)
        _check_missing(df, name, category, report_list)
        _check_datetime_format(df, name, category, report_list)  
                     
    if len(report_list) == 0: 
        logger.info(f" -> Không phát hiện lỗi cấu trúc")
    else: 
        audit_report.extend(report_list)
    
    return len(report_list)


### Kiểm tra logic nghiệp vụ ### 

def _check_product_margin(tables: dict, report_list: list, category: str):
    """
    Kiểm tra tính hợp lệ của dữ liệu định giá trong bảng products.

    Thực hiện 2 lớp kiểm tra theo thứ tự:
    1. Null check: Phát hiện sản phẩm thiếu giá vốn (cogs) hoặc giá bán (price).
       Các dòng null bị tách riêng trước để tránh pandas bỏ qua chúng trong phép so sánh.
    2. Margin check: Với các dòng có đủ data, kiểm tra cogs >= price (biên lợi nhuận âm).

    Args:
        tables (dict[str, pd.DataFrame]): Dictionary bảng đã load.
        report_list (list): Danh sách tích lũy issue của tầng Business Rules.
        category (str): Nhãn phân loại lỗi.

    Logs Issues:
        "{category}: Null Pricing"   — cogs hoặc price bị null.
        "{category}: Negative Margin" — cogs >= price (margin âm hoặc bằng 0).

    Skip condition:
        Hàm thoát sớm nếu "products" không có trong tables.
    """
    if "products" not in tables: return
    
    p = tables["products"]
    if "cogs" in p.columns and "price" in p.columns:
        # 1. Bắt lỗi thiếu giá (NaN)
        bad_null = p[p["cogs"].isna() | p["price"].isna()]
        if not bad_null.empty:
            indices = bad_null.index.tolist()[:5]
            log_issue(report_list, f"{category}: Null Pricing", "products", "cogs, price", 
                      "Thiếu dữ liệu Giá vốn hoặc Giá bán (NaN)", 
                      len(bad_null), f"Indices: {indices}", "Prep: Điền giá trị (Fill) hoặc Drop")

        # 2. So sánh Margin (Chỉ so sánh các dòng có đủ data để tránh bị Pandas lơ đi)
        p_valid = p.dropna(subset=["cogs", "price"])
        bad_cogs = p_valid[p_valid["cogs"] >= p_valid["price"]]
        
        if not bad_cogs.empty:
            indices = bad_cogs.index.tolist()[:5]
            log_issue(report_list, f"{category}: Negative Margin", "products", "cogs, price", 
                      "Giá vốn (COGS) >= Giá bán", 
                      len(bad_cogs), f"Indices: {indices}", "Kiểm tra lại biểu giá")

def _check_invalid_metrics(tables: dict, report_list: list, category: str):
    """
    Phát hiện các chỉ số vận hành có giá trị phi lý (<= 0 hoặc âm).

    Kiểm tra 3 chỉ số trên 3 bảng độc lập:
    1. order_items.quantity  <= 0 — Số lượng mua không thể bằng hoặc âm.
    2. returns.refund_amount < 0  — Tiền hoàn trả không thể âm.
    3. payments.payment_value <= 0 — Giá trị thanh toán phải dương.

    Mỗi bảng được kiểm tra độc lập — nếu bảng vắng mặt trong tables,
    check tương ứng sẽ bị skip mà không ảnh hưởng các check còn lại.

    Args:
        tables (dict[str, pd.DataFrame]): Dictionary bảng đã load.
        report_list (list): Danh sách tích lũy issue của tầng Business Rules.
        category (str): Nhãn phân loại lỗi.
    """
    # 1. Số lượng mua <= 0
    if "order_items" in tables:
        oi = tables["order_items"]
        if "quantity" in oi.columns:
            bad_qty = oi[oi["quantity"] <= 0]
            if not bad_qty.empty:
                indices = bad_qty.index.tolist()[:5]
                log_issue(report_list, category, "order_items", "quantity", 
                          "Số lượng mua <= 0 (Phi lý)", len(bad_qty), f"Indices: {indices}", "Drop dòng")

    # 2. Tiền hoàn trả < 0
    if "returns" in tables:
        r = tables["returns"]
        if "refund_amount" in r.columns:
            bad_refund = r[r["refund_amount"] < 0]
            if not bad_refund.empty:
                indices = bad_refund.index.tolist()[:5]
                log_issue(report_list, category, "returns", "refund_amount", 
                          "Tiền hoàn bị âm", len(bad_refund), f"Indices: {indices}", "Kiểm tra lại, cân nhắc lấy trị tuyệt đối (abs)")

    # 3. Thanh toán <= 0
    if "payments" in tables:
        pay = tables["payments"]
        if "payment_value" in pay.columns:
            bad_pay = pay[pay["payment_value"] <= 0]
            if not bad_pay.empty:
                indices = bad_pay.index.tolist()[:5]
                log_issue(report_list, category, "payments", "payment_value", 
                          "Thanh toán <= 0", len(bad_pay), f"Indices: {indices}", "Kiểm tra lại, drop dòng")

def _check_min_order_value_compliance(tables: dict, report_list: list, category: str):
    """
    Kiểm tra vi phạm ngưỡng giá trị đơn hàng tối thiểu (Minimum Order Value - MOV).

    Logic:
    - Tính order_total_value = sum(quantity * unit_price) theo từng order_id.
    - Join với bảng promotions để lấy min_order_value của mã chính (promo_id)
      và mã phụ (promo_id_2) nếu có.
    - min_order_value = NaN được fillna(0) — mã không yêu cầu MOV luôn hợp lệ.
    - Flag lỗi nếu order_total_value < min_order_value của bất kỳ mã nào được áp.

    Args:
        tables (dict[str, pd.DataFrame]): Dictionary bảng đã load.
                                          Cần có: "order_items", "promotions".
        report_list (list): Danh sách tích lũy issue của tầng Business Rules.
        category (str): Nhãn phân loại lỗi.

    Logs Issues:
        "{category}: MOV" trên cột "promo_id"   — mã chính vi phạm MOV.
        "{category}: MOV" trên cột "promo_id_2" — mã phụ vi phạm MOV.

    Skip condition:
        Hàm thoát sớm nếu thiếu "order_items" hoặc "promotions" trong tables.
    """
    req_tables = ["order_items", "promotions"]
    if not all(k in tables for k in req_tables):
        return

    df_oi = tables["order_items"].copy()
    
    # Bổ sung fillna(0) để xử lý các mã không yêu cầu đơn hàng tối thiểu
    df_promo = tables["promotions"][["promo_id", "min_order_value"]].copy()
    df_promo["min_order_value"] = df_promo["min_order_value"].fillna(0)

    # Tính tổng giá trị đơn hàng (trước chiết khấu)
    df_oi["line_total"] = df_oi["quantity"] * df_oi["unit_price"]
    df_order_totals = df_oi.groupby("order_id")["line_total"].sum().reset_index()
    df_order_totals.rename(columns={"line_total": "order_total_value"}, inplace=True)
    df_oi = df_oi.merge(df_order_totals, on="order_id", how="left")

    # Đối chiếu mã chính
    df_merged = df_oi.merge(
        df_promo.rename(columns={"promo_id": "p1_id", "min_order_value": "p1_min_val"}),
        left_on="promo_id", right_on="p1_id", how="left"
    )

    # Đối chiếu mã phụ
    if "promo_id_2" in df_merged.columns:
        df_merged = df_merged.merge(
            df_promo.rename(columns={"promo_id": "p2_id", "min_order_value": "p2_min_val"}),
            left_on="promo_id_2", right_on="p2_id", how="left"
        )
    else:
        df_merged["p2_min_val"] = 0.0    

    # Ghi nhận lỗi Mã chính
    p1_violation_mask = (df_merged["p1_id"].notna()) & (df_merged["order_total_value"] < df_merged["p1_min_val"])
    df_p1_violation = df_merged[p1_violation_mask]
    if not df_p1_violation.empty:
        error_orders_p1 = df_p1_violation["order_id"].unique().tolist()[:5]
        log_issue(report_list, f"{category}: MOV", "order_items", "promo_id", 
                  "Tổng đơn hàng không đạt mức tối thiểu của Mã chính", 
                  len(df_p1_violation), f"Order IDs: {error_orders_p1}", "Hướng xử lý: Xóa mã và tính lại")

    # Ghi nhận lỗi Mã phụ
    if "promo_id_2" in df_merged.columns:
        p2_violation_mask = (df_merged["p2_id"].notna()) & (df_merged["order_total_value"] < df_merged["p2_min_val"])
        df_p2_violation = df_merged[p2_violation_mask]
        if not df_p2_violation.empty:
            error_orders_p2 = df_p2_violation["order_id"].unique().tolist()[:5]
            log_issue(report_list, f"{category}: MOV", "order_items", "promo_id_2", 
                      "Tổng đơn hàng không đạt mức tối thiểu của Mã phụ", 
                      len(df_p2_violation), f"Order IDs: {error_orders_p2}", "Hướng xử lý: Xóa mã và tính lại")

def _check_stacked_promo_compliance(tables: dict, report_list: list, category: str):
    """
    Kiểm tra 2 loại vi phạm logic khuyến mãi: stackable rule và độ chính xác công thức.

    Lỗi 1 — Vi phạm Stackable Rule:
        Phát hiện dòng có promo_id_2 (mã phụ) trong khi promo_id (mã chính) có
        stackable_flag = 0. Đây là gian lận vì mã chính cấm cộng dồn nhưng mã phụ
        vẫn được áp. Gợi ý xử lý: xóa promo_id_2.

    Lỗi 2 — Sai lệch công thức (Combo Math):
        Tính expected_discount_amount dựa trên công thức từ data dictionary:
        - percentage: quantity × unit_price × (discount_value / 100)
        - fixed      : quantity × discount_value
        Áp dụng cho cả mã chính và mã phụ, cộng tổng lại thành expected_total.
        So sánh với discount_amount thực tế trong order_items bằng relative tolerance
        lấy từ config.MATH_TOLERANCE_PCT và config.MATH_TOLERANCE_MIN.

    Args:
        tables (dict[str, pd.DataFrame]): Dictionary bảng đã load.
                                          Cần có: "order_items", "promotions".
        report_list (list): Danh sách tích lũy issue của tầng Business Rules.
        category (str): Nhãn phân loại lỗi.

    Logs Issues:
        "{category}: Rule" — vi phạm stackable, cột "promo_id_2".
        "{category}: Math" — sai lệch discount_amount, cột "discount_amount".

    Skip condition:
        Hàm thoát sớm nếu thiếu "order_items" hoặc "promotions" trong tables.
    """
    req_tables = ["order_items", "promotions"]
    if not all(k in tables for k in req_tables):
        return

    df_oi = tables["order_items"].copy()
    df_promo = tables["promotions"][["promo_id", "promo_type", "discount_value", "stackable_flag"]]

    df_merged = df_oi.merge(
        df_promo.rename(columns={"promo_id": "p1_id", "promo_type": "p1_type", "discount_value": "p1_val", "stackable_flag": "p1_stack"}),
        left_on="promo_id", right_on="p1_id", how="left"
    )

    df_merged = df_merged.merge(
        df_promo.rename(columns={"promo_id": "p2_id", "promo_type": "p2_type", "discount_value": "p2_val", "stackable_flag": "p2_stack"}),
        left_on="promo_id_2", right_on="p2_id", how="left"
    )

    # Lỗi 1: Vi phạm quy tắc cộng dồn
    has_promo_2_mask = df_merged["p2_id"].notna() & (df_merged["p2_id"].astype(str).str.strip() != "") & (df_merged["p2_id"].astype(str).str.strip() != "0")
    abuse_mask = has_promo_2_mask & (df_merged["p1_stack"] == 0)
    df_abuse = df_merged[abuse_mask]
    
    if not df_abuse.empty:
        error_orders = df_abuse["order_id"].head(5).tolist()
        log_issue(report_list, f"{category}: Rule", "order_items", "promo_id_2", 
                  "Áp dụng mã phụ khi mã chính không cho phép cộng dồn", 
                  len(df_abuse), f"Order IDs: {error_orders}", "Hướng xử lý: Xóa bỏ mã phụ")

    # Lỗi 2: Sai lệch công thức
    d1_fixed = df_merged["quantity"] * df_merged["p1_val"]
    d1_pct = df_merged["quantity"] * df_merged["unit_price"] * (df_merged["p1_val"] / 100.0)
    df_merged["expected_d1"] = np.where(df_merged["p1_type"] == "fixed", d1_fixed, d1_pct)
    
    d2_fixed = df_merged["quantity"] * df_merged["p2_val"]
    d2_pct = df_merged["quantity"] * df_merged["unit_price"] * (df_merged["p2_val"] / 100.0)
    df_merged["expected_d2"] = np.where(df_merged["p2_type"] == "fixed", d2_fixed, d2_pct)

    df_merged["expected_total"] = df_merged["expected_d1"].fillna(0) + df_merged["expected_d2"].fillna(0)
    
    margin_of_error = df_merged["expected_total"].abs().clip(lower=config.MATH_TOLERANCE_MIN) * config.MATH_TOLERANCE_PCT
    bad_math_mask = abs(df_merged["discount_amount"].fillna(0) - df_merged["expected_total"]) > margin_of_error
    df_bad_math = df_merged[bad_math_mask]

    if not df_bad_math.empty:
        error_orders_math = df_bad_math["order_id"].head(5).tolist()
        log_issue(report_list, f"{category}: Math", "order_items", "discount_amount", 
                  "discount_amount không khớp với tổng chiết khấu dự kiến", 
                  len(df_bad_math), f"Order IDs: {error_orders_math}", "Hướng xử lý: Tính lại theo công thức")

def check_business_rules(tables: dict, audit_report: list, category: str = "Business_Rules"): 
    """
    Orchestrator kiểm định logic nghiệp vụ — chạy tuần tự 4 nhóm kiểm tra.

    Thứ tự thực thi:
    1. _check_product_margin          — biên lợi nhuận sản phẩm.
    2. _check_invalid_metrics         — chỉ số vận hành phi lý.
    3. _check_min_order_value_compliance — vi phạm MOV.
    4. _check_stacked_promo_compliance   — vi phạm stackable & sai công thức.

    Args:
        tables (dict[str, pd.DataFrame]): Dictionary bảng đã load từ load_data().
        audit_report (list): Danh sách tổng hợp toàn bộ issue. Kết quả sẽ được extend.
        category (str): Nhãn phân loại gốc, mặc định "Business_Rules".
                        Các hàm con có thể thêm sub-category (ví dụ: "Business_Rules: MOV").

    Returns:
        int: Số lượng issue phát hiện được trong tầng kiểm định này.
    """
    logger.info(" 2. Kiểm định logic nghiệp vụ")
    report_list = []

    _check_product_margin(tables, report_list, category)
    _check_invalid_metrics(tables, report_list, category)
    _check_min_order_value_compliance(tables, report_list, category)
    _check_stacked_promo_compliance(tables, report_list, category)

    if len(report_list) == 0: 
        logger.info(f" -> Không phát hiện lỗi logic nghiệp vụ")
    else: 
        audit_report.extend(report_list)

    return len(report_list)

### Kiểm định tính toàn vẹn tham chiếu ###
def check_referential(tables: dict, audit_report: list, category: str = "Referential"): 
    """
    Kiểm định tính toàn vẹn tham chiếu (Referential Integrity) theo danh sách FK_CHECKS.

    Với mỗi cặp (child_table.child_col → parent_table.parent_col) trong config.FK_CHECKS,
    hàm xác định các bản ghi "mồ côi" (orphan): giá trị không null trong child_col
    nhưng không tồn tại trong parent_col. Null trong child_col được bỏ qua
    (đã được xử lý bởi NULLABLE_BY_DESIGN trong _check_missing).

    Args:
        tables (dict[str, pd.DataFrame]): Dictionary bảng đã load từ load_data().
        audit_report (list): Danh sách tổng hợp toàn bộ issue. Kết quả sẽ được extend.
        category (str): Nhãn phân loại, mặc định "Referential".

    Returns:
        int: Số lượng issue phát hiện được trong tầng kiểm định này.

    Config dependency:
        config.FK_CHECKS (list[tuple]): Danh sách quan hệ FK cần kiểm tra, format:
        (child_table, child_col, parent_table, parent_col).
        Cặp nào thiếu bảng trong tables sẽ bị skip tự động.

    Logs Issue:
        Ghi nhận tên bảng cha trong Issue Description để dễ truy vết nguồn gốc orphan.
        Note: "Cần Drop hoặc gán 'Unknown'" tùy chiến lược preprocessing.
    """
    logger.info(" 3. Kiểm định tính toàn vẹn tham chiếu")
    report_list = []

    for child_tbl, child_col, parent_tbl, parent_col in config.FK_CHECKS:
        if child_tbl in tables and parent_tbl in tables:
            child_df = tables[child_tbl]
            parent_df = tables[parent_tbl]
            
            if child_col in child_df.columns and parent_col in parent_df.columns:
                not_null_mask = child_df[child_col].notna()
                orphans_mask = not_null_mask & (~child_df[child_col].isin(parent_df[parent_col]))
                n_orphans = orphans_mask.sum()
                
                if n_orphans > 0:
                    bad_indices = child_df[orphans_mask].index.tolist()[:5]
                    log_issue(report_list, category, child_tbl, child_col, 
                              f"Tham chiếu không tồn tại trong {parent_tbl}", 
                              n_orphans, f"Indices: {bad_indices}", "Cần Drop hoặc gán 'Unknown'")
                    
    if len(report_list) == 0: 
        logger.info(f" -> Không phát hiện lỗi tham chiếu")
    else: 
        audit_report.extend(report_list)
    
    return len(report_list)

### Đối soát tài chính ###

def check_reconciliation(tables: dict, audit_report: list, category: str = "Reconciliation"): 
    """
    Đối soát tài chính cấp vĩ mô (Macro-level) giữa order_items và bảng sales tổng hợp.

    Thực hiện 2 bước đối chiếu theo ngày:

    Bước 1 — Gross Revenue:
        Tính sum(quantity * unit_price) từ order_items, group by order_date.
        So sánh với cột Revenue trong bảng sales.
        Gross Revenue không trừ discount và không lọc returned orders —
        đây là convention của bảng sales được BTC cung cấp.

    Bước 2 — COGS:
        Tính sum(quantity * cogs) bằng cách join order_items với products,
        group by order_date. So sánh với cột COGS trong bảng sales.
        Bước 2 chỉ chạy nếu "products" có trong tables VÀ cột "COGS"
        tồn tại trong bảng sales.

    Tolerance: Sử dụng relative tolerance (config.MATH_TOLERANCE_PCT = 0.1%)
    với chặn dưới config.MATH_TOLERANCE_MIN = 1.0 VND để tránh false positive
    do sai lệch làm tròn float.

    Args:
        tables (dict[str, pd.DataFrame]): Dictionary bảng đã load từ load_data().
        audit_report (list): Danh sách tổng hợp toàn bộ issue. Kết quả sẽ được extend.
        category (str): Nhãn phân loại, mặc định "Reconciliation".

    Returns:
        int: Số lượng issue phát hiện được. Trả về 0 nếu thiếu bảng bắt buộc.

    Skip condition:
        Thoát sớm và trả về 0 nếu thiếu bất kỳ bảng nào trong:
        ["order_items", "orders", "sales"].

    Logs Issues:
        "{category}: Revenue Mismatch" — doanh thu ngày không khớp.
        "{category}: COGS Mismatch"    — giá vốn ngày không khớp.
    """
    logger.info(" 4. Đối soát tài chính ")
    report_list = []
    
    req_tables = ["order_items", "orders", "sales"]
    if not all(k in tables for k in req_tables):
        return

    df_oi = tables["order_items"].copy()
    df_ord = tables["orders"].copy()
    df_sales = tables["sales"].copy()

    # Đồng bộ định dạng chuẩn cho các cột ngày tháng để tránh lỗi khi gộp bảng
    df_ord["order_date"] = pd.to_datetime(df_ord["order_date"]).dt.date
    df_sales["Date"] = pd.to_datetime(df_sales["Date"]).dt.date

    # ==========================================
    # BƯỚC 1: ĐỐI SOÁT DOANH THU (GROSS REVENUE)
    # ==========================================
    # Tính Gross Revenue cho từng dòng sản phẩm
    df_oi["line_gross_revenue"] = df_oi["quantity"] * df_oi["unit_price"]
    
    # Kết nối với bảng orders để lấy ngày giao dịch
    df_merged_rev = df_oi.merge(df_ord[["order_id", "order_date"]], on="order_id", how="left")
    
    # Gom nhóm và tính tổng doanh thu theo từng ngày
    df_daily_rev = df_merged_rev.groupby("order_date")["line_gross_revenue"].sum().reset_index()
    df_daily_rev.rename(columns={"order_date": "Date"}, inplace=True)

    # Đối chiếu với bảng Sales của BTC
    df_val_rev = df_daily_rev.merge(df_sales[["Date", "Revenue"]], on="Date", how="inner")
    
    if "Revenue" in df_val_rev.columns:
        # Biên độ sai số (Margin of error) = 1.0 để bù trừ sai lệch do làm tròn số thập phân (float)
        margin_rev = df_val_rev["line_gross_revenue"].abs().clip(lower=config.MATH_TOLERANCE_MIN) * config.MATH_TOLERANCE_PCT
        gross_diff_mask = abs(df_val_rev["line_gross_revenue"] - df_val_rev["Revenue"]) > margin_rev
        df_gross_diff = df_val_rev[gross_diff_mask]
        
        if not df_gross_diff.empty:
            bad_dates_gross = df_gross_diff["Date"].head(5).astype(str).tolist()
            log_issue(
                report_list, f"{category}: Revenue Mismatch", "sales", "Revenue",
                "Gross Revenue tổng hợp từ chi tiết không khớp với báo cáo sales",
                len(df_gross_diff), f"Dates: {bad_dates_gross}", "Cần kiểm tra lại các giao dịch bất thường trong ngày"
            )

    # ==========================================
    # BƯỚC 2: ĐỐI SOÁT GIÁ VỐN HÀNG BÁN (COGS)
    # ==========================================
    if "products" in tables and "COGS" in df_sales.columns: 
        df_prod = tables["products"].copy()
        
        # Kết nối lấy giá vốn gốc từ danh mục sản phẩm
        df_oi_cogs = df_oi.merge(df_prod[["product_id", "cogs"]], on="product_id", how="left")
        df_oi_cogs["line_cogs"] = df_oi_cogs["quantity"] * df_oi_cogs["cogs"].fillna(0)
        
        # Kết nối lấy ngày và gom nhóm theo ngày
        df_merged_cogs = df_oi_cogs.merge(df_ord[["order_id", "order_date"]], on="order_id", how="left")
        df_daily_cogs = df_merged_cogs.groupby("order_date")["line_cogs"].sum().reset_index()
        df_daily_cogs.rename(columns={"order_date": "Date"}, inplace=True)
        
        # Đối chiếu với bảng Sales
        df_val_cogs = df_daily_cogs.merge(df_sales[["Date", "COGS"]], on="Date", how="inner")

        margin_cogs = df_val_cogs["line_cogs"].abs().clip(lower=config.MATH_TOLERANCE_MIN) * config.MATH_TOLERANCE_PCT
        cogs_diff_mask = abs(df_val_cogs["line_cogs"] - df_val_cogs["COGS"]) > margin_cogs
        df_cogs_diff = df_val_cogs[cogs_diff_mask]
        
        if not df_cogs_diff.empty:
            bad_dates_cogs = df_cogs_diff["Date"].head(5).astype(str).tolist()
            log_issue(
                report_list, f"{category}: COGS Mismatch", "sales", "COGS",
                "Tổng giá vốn (COGS) từ chi tiết không khớp với báo cáo sales",
                len(df_cogs_diff), f"Dates: {bad_dates_cogs}", "Cần rà soát lại dữ liệu trống trong bảng products"
            )

    if len(report_list) == 0: 
        logger.info(f" -> Đối soát tài chính thành công.")
    else: 
        audit_report.extend(report_list)
    
    return len(report_list)

def run_audit(): 
    logger.info("Data Audit ...")
    tables = load_data(config.SOURCE_TABLES) 
    audit_report = [] 

    s  = check_structural(tables, audit_report, config.SOURCE_TABLES)
    b  = check_business_rules(tables, audit_report)
    r  = check_referential(tables, audit_report) 
    rc = check_reconciliation(tables, audit_report)

    df = pd.DataFrame(audit_report)
    if len(df) > 0:
        output_path = config.PROCESSED / "qa_report.csv"
        df.to_csv(output_path, index=False, encoding="utf-8-sig")
        logger.info(f"[SUMMARY] Structural: {s} | Business: {b} | Referential: {r} | Reconciliation: {rc} lỗi")
        logger.info(f"Phát hiện {len(df)} lỗi. Lưu tại {output_path}")
    else: 
        logger.info("Dữ liệu sạch 100%")