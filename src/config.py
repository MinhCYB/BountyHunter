import os  
import logging
from pathlib import Path 

PROJECT_ROOT = Path(__file__).resolve().parents[1] 

class Config: 
    # ==========================================
    # FILE, FOLDER PATH
    # ==========================================
    RAW = PROJECT_ROOT / "data/raw"
    PROCESSED = PROJECT_ROOT / "data/processed"
    FEATURES = PROJECT_ROOT / "data/features" 
    SUBMISSIONS = PROJECT_ROOT / "submissions"
    REPORTS = PROJECT_ROOT / "reports"
    LOGS = PROJECT_ROOT / "logs"
    LOG_FILE_PATH = LOGS / "BountyHunter.log"


    # ==========================================
    # MODEL CONFIG
    # ==========================================
    RANDOM_STATE = 42 
    PARAMS = {
        # Thêm sau    
    }

    # ==========================================
    # DATE SPINE CONFIGURATION (TRỤC THỜI GIAN)
    # ==========================================
    DATE_SPINE_START = '2012-07-04'
    DATE_SPINE_END = '2024-07-01'

    # Prep & audit
    SOURCE_TABLES = [
        "customers", "geography", "inventory", "order_items", 
        "orders", "payments", "products", "promotions", 
        "returns", "reviews", "sales", "shipments", "web_traffic"
    ]

    DATE_COLS = {
        "orders"      : ["order_date"],
        "shipments"   : ["ship_date", "delivery_date"],
        "returns"     : ["return_date"],
        "reviews"     : ["review_date"],
        "customers"   : ["signup_date"],
        "promotions"  : ["start_date", "end_date"],
        "inventory"   : ["snapshot_date"],
        "web_traffic" : ["date"],
        "sales"       : ["Date"],
    }

    # Không phải bug khi null
    NULLABLE_BY_DESIGN = {
        "customers"  : ["gender", "age_group", "acquisition_channel"],
        "promotions" : ["applicable_category", "promo_channel", "min_order_value"],
        "order_items": ["promo_id", "promo_id_2"],
    }

    FK_CHECKS = [
        # Cấu trúc: (bảng_con, cột_con, bảng_cha, cột_cha)

        # 1. Nhóm bám theo Đơn hàng (Orders) - Bảng trung tâm
        ("order_items", "order_id",    "orders",      "order_id"),
        ("shipments",   "order_id",    "orders",      "order_id"),
        ("payments",    "order_id",    "orders",      "order_id"),
        ("returns",     "order_id",    "orders",      "order_id"),
        ("reviews",     "order_id",    "orders",      "order_id"),

        # 2. Nhóm bám theo Sản phẩm (Products)
        ("order_items", "product_id",  "products",    "product_id"),
        ("inventory",   "product_id",  "products",    "product_id"),
        ("returns",     "product_id",  "products",    "product_id"),
        ("reviews",     "product_id",  "products",    "product_id"),

        # 3. Nhóm bám theo Khách hàng (Customers)
        ("orders",      "customer_id", "customers",   "customer_id"),
        ("reviews",     "customer_id", "customers",   "customer_id"),

        # 4. Nhóm bám theo Khuyến mãi (Promotions)
        ("order_items", "promo_id",    "promotions",  "promo_id"),
        ("order_items", "promo_id_2",  "promotions",  "promo_id"),

        # 5. Nhóm bám theo Địa lý (Geography)
        ("customers",   "zip",         "geography",   "zip"),
        ("orders",      "zip",         "geography",   "zip"),
    ]

    # ==========================================
    # AUDIT TOLERANCE THRESHOLDS (NGƯỠNG SAI SỐ)
    # ==========================================
    MATH_TOLERANCE_PCT = 0.001  # Sai số tương đối (0.1%)
    MATH_TOLERANCE_MIN = 1.0    # Chặn dưới sai số tối thiểu (1.0 VND)

    # ==========================================
    # SCHEMA METADATA (DATA QUALITY)
    # ==========================================
    # Cột ID kiểu string cần chuẩn hóa — tránh Silent Merge Failure khi pandas
    # tự ép NaN-mixed column sang float64 (vd: "123" → "123.0" → join miss).
    STRING_ID_COLS = {"promo_id", "promo_id_2", "return_id", "review_id"}
 
    # Cột hằng số xác nhận từ profiling report (Unique=1) — drop để giảm noise
    CONSTANT_COLS = {"reorder_flag"}

    def __init__(self): 
        self.LOGS.mkdir(parents=True, exist_ok=True) 
        with open(self.LOG_FILE_PATH, "w", encoding="utf-8"):
            pass

    def get_logger(self, name): 
        logger = logging.getLogger(name)

        if not logger.handlers: 
            logger.setLevel(logging.INFO) 
            formatter = logging.Formatter(
                "[%(name)s][%(levelname)s]: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            file_handler = logging.FileHandler(self.LOG_FILE_PATH, encoding="utf-8")
            
            file_handler.setLevel(logging.INFO) 
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger

config = Config()