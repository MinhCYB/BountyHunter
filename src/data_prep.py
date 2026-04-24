"""
Docstring for data_prep
Gộp data 
format lại data


"""

from src.config import config 

import pandas as pd 
import numpy as np  

def cast_datetime_columns(name: str, df: pd.DataFrame) -> pd.DataFrame:
    if name in config.DATE_COLS:
        cols_to_cast = config.DATE_COLS[name]
        
        for col in cols_to_cast:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                print(f" {name}['{col}'] -> datetime ok")
            else:
                print(f"  [WARNING] Cột '{col}' không có trong '{name}.csv'")
                
    return df

def format_and_save_data(source_tables: list):
    """
    Docstring for format_and_save_data
    Cast dữ liệu sang datetime và lưu dạng parquet
    """
    
    for csv_file in sorted(config.RAW.glob("*.csv")):
        table_name = csv_file.stem
        
        if table_name in source_tables:
            df = pd.read_csv(csv_file, low_memory=False)
            df = cast_datetime_columns(name=table_name, df=df)
            df.to_parquet(config.PROCESSED / f"{table_name}.parquet", index=False)
            print(f"-> Đã tải và chuẩn hoá bảng: {table_name}")

# ==========================================
# CÁC HÀM XỬ LÝ TỪNG CỤM (CLUSTERS)
# ==========================================

def _prep_transaction_cluster() -> pd.DataFrame:
    """Cụm 1: Giao dịch & Sản phẩm"""
    print("  -> Đang xử lý Cụm Giao dịch...")
    orders = pd.read_parquet(config.PROCESSED / "orders.parquet")
    order_items = pd.read_parquet(config.PROCESSED / "order_items.parquet")
    
    # Đưa về 00:00:00
    orders['order_date'] = orders['order_date'].dt.normalize()
    
    # 1. Tính tổng giá trị đơn hàng từ order_items
    item_agg = order_items.groupby('order_id').agg(
        total_items_sold=('quantity', 'sum'),
        total_discount=('discount_amount', 'sum')
    ).reset_index()
    
    # 2. Ráp vào orders và gom nhóm theo NGÀY
    orders_full = orders.merge(item_agg, on='order_id', how='left')
    
    daily_tx = orders_full.groupby('order_date').agg(
        total_orders=('order_id', 'nunique'),
        total_items_sold=('total_items_sold', 'sum'),
        total_discount=('total_discount', 'sum')
    ).reset_index() 
    
    daily_tx = daily_tx.rename(columns={'order_date': 'Date'})
    return daily_tx

def _prep_customer_cluster() -> pd.DataFrame:
    """Cụm 2: Khách hàng (Returns & Reviews)"""
    print("  -> Đang xử lý Cụm Khách hàng...")
    returns = pd.read_parquet(config.PROCESSED / "returns.parquet")
    reviews = pd.read_parquet(config.PROCESSED / "reviews.parquet")
    
    # Xử lý Returns (Gom theo ngày trả hàng)
    returns['return_date'] = pd.to_datetime(returns['return_date']).dt.normalize()
    daily_returns = returns.groupby('return_date').agg(
        total_returns=('return_id', 'nunique'),
        return_loss_amount=('refund_amount', 'sum')
    ).reset_index().rename(columns={'return_date': 'Date'})
    
    # Xử lý Reviews (Gom theo ngày đánh giá)
    reviews['review_date'] = pd.to_datetime(reviews['review_date']).dt.normalize()
    daily_reviews = reviews.groupby('review_date').agg(
        avg_rating=('rating', 'mean'),
        total_reviews=('review_id', 'count')
    ).reset_index().rename(columns={'review_date': 'Date'})
    
    # Outer merge để ghép 2 bảng này thành 1 (vì có ngày có review nhưng ko có return)
    daily_customer = pd.merge(daily_returns, daily_reviews, on='Date', how='outer').fillna(0)
    return daily_customer

def _prep_operation_cluster(date_spine: pd.DataFrame) -> pd.DataFrame:
    """Cụm 3: Vận hành (Web Traffic & Inventory)"""
    print("  -> Đang xử lý Cụm Vận hành...")
    web_traffic = pd.read_parquet(config.PROCESSED / "web_traffic.parquet")
    inventory = pd.read_parquet(config.PROCESSED / "inventory.parquet")
    
    # 1. Traffic đã theo ngày, chỉ cần gom nhóm cho chắc chắn
    web_traffic['date'] = pd.to_datetime(web_traffic['date']).dt.normalize()
    daily_traffic = web_traffic.groupby('date').agg(
        total_sessions=('sessions', 'sum'),
        total_visitors=('unique_visitors', 'sum'),
        avg_bounce_rate=('bounce_rate', 'mean')
    ).reset_index().rename(columns={'date': 'Date'})
    
    # 2. GIẢI QUYẾT INVENTORY (Tháng -> Ngày)
    # Lấy trung bình tồn kho của tháng đó
    inventory['snapshot_date'] = pd.to_datetime(inventory['snapshot_date'])
    inventory['YearMonth'] = inventory['snapshot_date'].dt.to_period('M')
    
    monthly_inv = inventory.groupby('YearMonth').agg(
        avg_fill_rate=('fill_rate', 'mean'),
        total_stockout_days=('stockout_days', 'sum')
    ).reset_index()
    
    # Tạo khóa YearMonth cho Date Spine để left join (Forward fill xịn)
    temp_spine = date_spine[['Date']].copy()
    temp_spine['YearMonth'] = temp_spine['Date'].dt.to_period('M')
    daily_inv = temp_spine.merge(monthly_inv, on='YearMonth', how='left').drop(columns=['YearMonth'])
    
    # Ghép Traffic và Inventory lại
    daily_ops = pd.merge(daily_traffic, daily_inv, on='Date', how='outer')
    return daily_ops

# ==========================================
# HÀM CHÍNH: LẮP RÁP MASTER TABLE
# ==========================================

def build_base_table(): 
    """Gom nhóm dữ liệu thành bảng cơ sở cho training model"""
    print("=======================================")
    print("Bắt đầu xây dựng Base Table (Full Mode)...")
    
    # 1. Tạo "Xương sống" (Từ ngày nhỏ nhất của tập Train đến ngày lớn nhất của tập Test)
    # Target: 01/01/2023 - 01/07/2024. Train: 04/07/2012 - 31/12/2022
    date_spine = pd.DataFrame({
        'Date': pd.date_range(start='2012-07-04', end='2024-07-01', freq='D')
    })
    
    # 2. Nạp target (sales.csv)
    sales = pd.read_parquet(config.PROCESSED / "sales.parquet")
    sales['Date'] = pd.to_datetime(sales['Date']).dt.normalize()
    base_table = date_spine.merge(sales, on='Date', how='left')
    
    # 3. Lắp ráp các module
    tx_df = _prep_transaction_cluster()
    base_table = base_table.merge(tx_df, on='Date', how='left')
    
    cus_df = _prep_customer_cluster()
    base_table = base_table.merge(cus_df, on='Date', how='left')
    
    ops_df = _prep_operation_cluster(date_spine)
    base_table = base_table.merge(ops_df, on='Date', how='left')
    
    # 4. Fill NaN an toàn (Không fill bằng 0 cho các biến có ý nghĩa đặc biệt)
    # Ví dụ: không có đơn = 0, không có review = 0
    cols_to_fill_0 = ['total_orders', 'total_items_sold', 'total_returns', 'total_reviews', 'total_sessions']
    for col in cols_to_fill_0:
        if col in base_table.columns:
            base_table[col] = base_table[col].fillna(0)
            
    # Lưu file ra chờ file feature_eng.py xử lý tiếp
    output_path = config.PROCESSED / "base_table.parquet"
    base_table.to_parquet(output_path, index=False)
    base_table.to_csv(config.PROCESSED / "base_table.csv", index=False)

    print(f"-> HOÀN TẤT! Master Table có kích thước: {base_table.shape}")
    print(f"-> Đã lưu tại: {output_path}")
    print("=======================================")


def run_prep(): 
    print(config.RAW)

    format_and_save_data(config.SOURCE_TABLES)

    build_base_table()
    