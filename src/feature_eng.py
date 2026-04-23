"""
feature_eng.py  —  Tạo toàn bộ derived features cho BountyHunter
Output: data/features/ (mỗi nhóm lưu 1 file parquet riêng)
Chạy:  python src/feature_eng.py

Nhóm features:
  1. Order-level time features          → order_time_features.parquet
  2. Order-line margin & discount        → orderline_features.parquet
  3. Customer-level features             → customer_features.parquet
  4. Cohort retention features           → cohort_retention.parquet
  5. Return-rate features                → return_features.parquet
                                         return_rate_by_category.parquet
                                         return_rate_by_size.parquet
  6. Daily conversion & traffic          → daily_features.parquet
                                         traffic_by_source.parquet
  7. Inventory loss features             → inventory_features.parquet
  8. Promo active flag per day           → promo_daily_flags.parquet
  9. Channel quality (Story 2a)          → channel_quality.parquet
 10. New vs Returning revenue per year   → new_vs_returning.parquet
 11. Campaign before/during/after        → campaign_windows.parquet
 12. Seasonal monthly index (Story 1b)   → seasonal_index.parquet
 13. Master feature table for model      → master_daily.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
import time
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────────────────

PROC_DIR = Path("data/processed")
FEAT_DIR = Path("data/features")
FEAT_DIR.mkdir(parents=True, exist_ok=True)

def section(t): print(f"\n{'='*60}\n  {t}\n{'='*60}")
def ok(msg):    print(f"  ok   {msg}")
def warn(msg):  print(f"  WARN {msg}")

def load(name):
    path = PROC_DIR / f"{name}.parquet"
    df = pd.read_parquet(path)
    ok(f"{name:<20} {df.shape}")
    return df

def save(df, name):
    path = FEAT_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    ok(f"saved → {path.name}  shape={df.shape}")
    return df

t0 = time.time()

# ─────────────────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────────────────

section("LOADING PROCESSED FILES")

orders      = load("orders")
order_items = load("order_items")
products    = load("products")
customers   = load("customers")
promotions  = load("promotions")
shipments   = load("shipments")
returns     = load("returns")
reviews     = load("reviews")
inventory   = load("inventory")
web_traffic = load("web_traffic")
sales       = load("sales")

# Chuẩn hoá tên cột sales về chữ thường
sales.columns = [c.lower() for c in sales.columns]   # Date→date, Revenue→revenue, COGS→cogs

# ─────────────────────────────────────────────────────────
# NHÓM 1 — Order-level time features
# Nguồn: orders
# Dùng cho: Story 1, model (Part 3)
# ─────────────────────────────────────────────────────────

section("NHÓM 1 — Order-level time features")

ord_feat = orders[["order_id", "customer_id", "order_date", "order_status"]].copy()

ord_feat["order_year"]      = ord_feat["order_date"].dt.year
ord_feat["order_month"]     = ord_feat["order_date"].dt.month
ord_feat["order_quarter"]   = ord_feat["order_date"].dt.quarter
ord_feat["order_dayofweek"] = ord_feat["order_date"].dt.dayofweek   # 0=Mon, 6=Sun
ord_feat["is_weekend"]      = ord_feat["order_dayofweek"].isin([5, 6]).astype(int)
ord_feat["is_month_end"]    = ord_feat["order_date"].dt.is_month_end.astype(int)
ord_feat["is_quarter_end"]  = ord_feat["order_date"].dt.is_quarter_end.astype(int)
ord_feat["is_year_end"]     = (
    (ord_feat["order_month"] == 12) & (ord_feat["order_date"].dt.day == 31)
).astype(int)

# Tết period: ±7 ngày quanh mùng 1 Tết âm lịch (dương lịch)
TET_DATES = {
    2012: "2012-01-23", 2013: "2013-02-10", 2014: "2014-01-31",
    2015: "2015-02-19", 2016: "2016-02-08", 2017: "2017-01-28",
    2018: "2018-02-16", 2019: "2019-02-05", 2020: "2020-01-25",
    2021: "2021-02-12", 2022: "2022-02-01",
}
tet_ranges = [
    (pd.Timestamp(d) - pd.Timedelta(days=7), pd.Timestamp(d) + pd.Timedelta(days=7))
    for d in TET_DATES.values()
]

def is_tet(dt):
    for start, end in tet_ranges:
        if start <= dt <= end:
            return 1
    return 0

ord_feat["is_tet_period"] = ord_feat["order_date"].map(is_tet)

# 11/11 Singles Day & 12/12
ord_feat["is_singles_day"] = (
    (ord_feat["order_month"] == 11) & (ord_feat["order_date"].dt.day == 11)
).astype(int)
ord_feat["is_double_12"] = (
    (ord_feat["order_month"] == 12) & (ord_feat["order_date"].dt.day == 12)
).astype(int)

save(ord_feat, "order_time_features")
ok(f"new cols: {[c for c in ord_feat.columns if c not in ['order_id','customer_id','order_date','order_status']]}")

# ─────────────────────────────────────────────────────────
# NHÓM 2 — Order-line margin & discount features
# Nguồn: order_items + products
# Dùng cho: Story 2, 3, 4
# ─────────────────────────────────────────────────────────

section("NHÓM 2 — Order-line margin & discount features")

oi = order_items.merge(
    products[["product_id", "price", "cogs", "category", "segment", "size"]],
    on="product_id", how="left"
)

oi["revenue_line"]      = oi["quantity"] * oi["unit_price"]
oi["cogs_line"]         = oi["quantity"] * oi["cogs"]
oi["gross_margin_line"] = oi["revenue_line"] - oi["cogs_line"]
oi["margin_rate_line"]  = (
    oi["gross_margin_line"] / oi["revenue_line"].replace(0, np.nan)
)

# Discount rate thực tế so với giá gốc
full_price = oi["quantity"] * oi["price"]
oi["discount_rate_actual"] = (
    oi["discount_amount"] / full_price.replace(0, np.nan)
).clip(0, 1)

oi["has_promo"]     = oi["promo_id"].notna().astype(int)
oi["is_discounted"] = (oi["discount_amount"] > 0).astype(int)

# AOV proxy: revenue / quantity
oi["revenue_per_unit"] = oi["revenue_line"] / oi["quantity"].replace(0, np.nan)

cols = [
    "order_id", "product_id", "quantity", "unit_price",
    "discount_amount", "promo_id",
    "category", "segment", "size",
    "revenue_line", "cogs_line", "gross_margin_line",
    "margin_rate_line", "discount_rate_actual",
    "has_promo", "is_discounted", "revenue_per_unit",
]
oi_feat = oi[cols].copy()

save(oi_feat, "orderline_features")
ok(f"margin_rate_line  median={oi_feat['margin_rate_line'].median():.3f}  mean={oi_feat['margin_rate_line'].mean():.3f}")
ok(f"has_promo rate    {oi_feat['has_promo'].mean()*100:.1f}%")
ok(f"discount_rate_actual  median={oi_feat['discount_rate_actual'].median():.3f}  p99={oi_feat['discount_rate_actual'].quantile(0.99):.3f}")

# ─────────────────────────────────────────────────────────
# NHÓM 3 — Customer-level features
# Nguồn: customers + orders + order_items
# Dùng cho: Story 2 (channel quality, cohort, new vs returning)
#
# NOTE: signup_date không đáng tin (89.3% khách có first_order TRƯỚC signup).
#       Dùng first_order_date làm activation_date thay thế.
# ─────────────────────────────────────────────────────────

section("NHÓM 3 — Customer-level features")

cust_orders = (
    orders[["order_id", "customer_id", "order_date", "order_status",
            "order_source", "device_type"]]
    .sort_values(["customer_id", "order_date"])
    .copy()
)

# --- Order count & repeat buyer ---
order_counts = (
    cust_orders.groupby("customer_id")["order_id"]
    .count().reset_index()
    .rename(columns={"order_id": "order_count"})
)
order_counts["is_repeat_buyer"] = (order_counts["order_count"] > 1).astype(int)

# --- First & last order date, tenure ---
order_range = (
    cust_orders.groupby("customer_id")["order_date"]
    .agg(first_order_date="min", last_order_date="max")
    .reset_index()
)
order_range["customer_tenure_days"] = (
    order_range["last_order_date"] - order_range["first_order_date"]
).dt.days

# --- Inter-order gap: median ngày giữa 2 lần mua liên tiếp ---
cust_orders["prev_order_date"] = (
    cust_orders.groupby("customer_id")["order_date"].shift(1)
)
cust_orders["inter_order_gap"] = (
    (cust_orders["order_date"] - cust_orders["prev_order_date"]).dt.days
)
median_gap = (
    cust_orders.dropna(subset=["inter_order_gap"])
    .groupby("customer_id")["inter_order_gap"]
    .median().reset_index()
    .rename(columns={"inter_order_gap": "median_inter_order_gap"})
)

# --- Revenue per customer ---
# Join order_items để lấy revenue
oi_rev = oi_feat[["order_id", "revenue_line", "gross_margin_line"]].copy()
order_revenue = (
    oi_rev.groupby("order_id")
    .agg(order_revenue=("revenue_line", "sum"),
         order_margin=("gross_margin_line", "sum"))
    .reset_index()
)
cust_revenue = (
    cust_orders[["order_id", "customer_id"]].merge(order_revenue, on="order_id", how="left")
    .groupby("customer_id")
    .agg(total_revenue=("order_revenue", "sum"),
         total_margin=("order_margin", "sum"))
    .reset_index()
)

# --- Retention: % khách có ≥2 đơn trong 6 tháng đầu kể từ first_order ---
# Lấy first_order_date, rồi đếm order trong [first+1day, first+180day]
first_orders = order_range[["customer_id", "first_order_date"]].copy()
early_orders = (
    cust_orders.merge(first_orders, on="customer_id")
    .assign(days_since_first=lambda x: (x["order_date"] - x["first_order_date"]).dt.days)
)
early_repeat = (
    early_orders[(early_orders["days_since_first"] > 0) &
                 (early_orders["days_since_first"] <= 180)]
    .groupby("customer_id")["order_id"].count().reset_index()
    .rename(columns={"order_id": "orders_in_first_6m"})
)
early_repeat["retained_6m"] = 1

# --- Gộp tất cả ---
cust_feat = (
    customers[["customer_id", "city", "gender", "age_group", "acquisition_channel"]]
    .merge(order_counts,  on="customer_id", how="left")
    .merge(order_range,   on="customer_id", how="left")
    .merge(median_gap,    on="customer_id", how="left")
    .merge(cust_revenue,  on="customer_id", how="left")
    .merge(early_repeat[["customer_id", "orders_in_first_6m", "retained_6m"]],
           on="customer_id", how="left")
)

cust_feat["order_count"]         = cust_feat["order_count"].fillna(0).astype(int)
cust_feat["is_repeat_buyer"]     = cust_feat["is_repeat_buyer"].fillna(0).astype(int)
cust_feat["retained_6m"]         = cust_feat["retained_6m"].fillna(0).astype(int)
cust_feat["orders_in_first_6m"]  = cust_feat["orders_in_first_6m"].fillna(0).astype(int)
cust_feat["total_revenue"]       = cust_feat["total_revenue"].fillna(0)
cust_feat["total_margin"]        = cust_feat["total_margin"].fillna(0)

# Revenue per customer (CLV proxy)
cust_feat["revenue_per_order"] = (
    cust_feat["total_revenue"] / cust_feat["order_count"].replace(0, np.nan)
)

save(cust_feat, "customer_features")
ok(f"repeat buyer rate    {cust_feat['is_repeat_buyer'].mean()*100:.1f}%")
ok(f"retained_6m rate     {cust_feat['retained_6m'].mean()*100:.1f}%")
ok(f"median inter_order_gap (repeat buyers): {cust_feat['median_inter_order_gap'].median():.0f} ngày")
ok(f"avg revenue_per_customer: {cust_feat['total_revenue'].mean():,.0f}")

# ─────────────────────────────────────────────────────────
# NHÓM 4 — Cohort retention heatmap data
# Nguồn: orders + order_items
# Dùng cho: Story 2b
# cohort = tháng đặt đơn đầu tiên
# retention_rate_month_n = % khách còn mua ở tháng thứ n
# ─────────────────────────────────────────────────────────

section("NHÓM 4 — Cohort retention features")

cohort_base = (
    cust_orders.groupby("customer_id")["order_date"]
    .min().reset_index()
    .rename(columns={"order_date": "cohort_date"})
)
cohort_base["cohort_month"] = cohort_base["cohort_date"].dt.to_period("M")

cohort_orders = (
    cust_orders[["customer_id", "order_date"]]
    .merge(cohort_base[["customer_id", "cohort_month"]], on="customer_id")
)
cohort_orders["order_period"]    = cohort_orders["order_date"].dt.to_period("M")
cohort_orders["months_since_first"] = (
    (cohort_orders["order_period"] - cohort_orders["cohort_month"])
    .apply(lambda x: x.n)
)

# Fix edge-of-month bias: cohort tháng nào cũng dùng toàn bộ tháng đó làm month_0
# Loại bỏ cohort có <10 khách (quá nhỏ, retention rate không ổn định)
cohort_size = (
    cohort_base.groupby("cohort_month")["customer_id"]
    .count().reset_index()
    .rename(columns={"customer_id": "cohort_size"})
)
valid_cohorts = cohort_size[cohort_size["cohort_size"] >= 10]["cohort_month"]

# Unique buyers per cohort per month_n
cohort_retention = (
    cohort_orders[
        (cohort_orders["months_since_first"] >= 0) &
        (cohort_orders["cohort_month"].isin(valid_cohorts))
    ]
    .groupby(["cohort_month", "months_since_first"])["customer_id"]
    .nunique().reset_index()
    .rename(columns={"customer_id": "active_customers"})
    .merge(cohort_size, on="cohort_month")
)
cohort_retention["retention_rate"] = (
    cohort_retention["active_customers"] / cohort_retention["cohort_size"]
)

# Giới hạn 12 tháng đầu để heatmap gọn
cohort_retention = cohort_retention[cohort_retention["months_since_first"] <= 12]

save(cohort_retention, "cohort_retention")
ok(f"cohorts: {cohort_retention['cohort_month'].nunique()}")
ok(f"month_0 retention = 1.0 (baseline), month_1 avg = {cohort_retention[cohort_retention['months_since_first']==1]['retention_rate'].mean():.3f}")

# ─────────────────────────────────────────────────────────
# NHÓM 5 — Return-rate features
# Nguồn: returns + order_items + products
# Dùng cho: Story 4b
# ─────────────────────────────────────────────────────────

section("NHÓM 5 — Return-rate features")

# Units sold / returned per product
units_sold = (
    order_items.groupby("product_id")["quantity"].sum()
    .reset_index().rename(columns={"quantity": "total_units_sold"})
)
units_returned = (
    returns.groupby("product_id")["return_quantity"].sum()
    .reset_index().rename(columns={"return_quantity": "total_units_returned"})
)

prod_return = units_sold.merge(units_returned, on="product_id", how="left")
prod_return["total_units_returned"] = prod_return["total_units_returned"].fillna(0)
prod_return["return_rate_product"]  = (
    prod_return["total_units_returned"] / prod_return["total_units_sold"]
).clip(0, 1)

# Revenue & margin per product
prod_revenue = (
    oi_feat.groupby("product_id")
    .agg(total_revenue=("revenue_line", "sum"),
         total_gross_margin=("gross_margin_line", "sum"))
    .reset_index()
)
refund_per_prod = (
    returns.groupby("product_id")["refund_amount"].sum()
    .reset_index().rename(columns={"refund_amount": "total_refund"})
)

prod_net = (
    prod_revenue
    .merge(refund_per_prod, on="product_id", how="left")
)
prod_net["total_refund"]                = prod_net["total_refund"].fillna(0)
prod_net["net_margin_after_return"]     = prod_net["total_gross_margin"] - prod_net["total_refund"]
prod_net["net_margin_rate_after_return"] = (
    prod_net["net_margin_after_return"] / prod_net["total_revenue"].replace(0, np.nan)
)

# Return reason distribution per product
return_reason_dist = (
    returns.groupby(["product_id", "return_reason"])["return_id"]
    .count().reset_index().rename(columns={"return_id": "reason_count"})
)
top_reason = (
    return_reason_dist.sort_values("reason_count", ascending=False)
    .drop_duplicates("product_id")[["product_id", "return_reason"]]
    .rename(columns={"return_reason": "top_return_reason"})
)

# Gộp tất cả product-level return features
return_feat = (
    prod_return
    .merge(prod_net[["product_id", "net_margin_after_return",
                      "net_margin_rate_after_return"]], on="product_id", how="left")
    .merge(products[["product_id", "category", "segment", "size",
                      "price", "cogs"]], on="product_id", how="left")
    .merge(top_reason, on="product_id", how="left")
)
return_feat["margin_rate"] = (
    (return_feat["price"] - return_feat["cogs"]) / return_feat["price"]
)

# Quadrant classification cho Story 4b
margin_median = return_feat["margin_rate"].median()
return_median = return_feat["return_rate_product"].median()

def quadrant(row):
    hi_m = row["margin_rate"]    >= margin_median
    hi_r = row["return_rate_product"] >= return_median
    if hi_m and not hi_r:   return "High_Margin_Low_Return"    # ideal
    if hi_m and hi_r:        return "High_Margin_High_Return"   # fix returns
    if not hi_m and hi_r:    return "Low_Margin_High_Return"    # consider drop
    return                          "Low_Margin_Low_Return"      # volume play

return_feat["product_quadrant"] = return_feat.apply(quadrant, axis=1)

# Return rate by category
cat_sold = (
    order_items.merge(products[["product_id", "category"]], on="product_id")
    .groupby("category")["quantity"].sum()
    .reset_index().rename(columns={"quantity": "cat_units_sold"})
)
cat_ret = (
    returns.merge(products[["product_id", "category"]], on="product_id")
    .groupby("category")["return_quantity"].sum()
    .reset_index().rename(columns={"return_quantity": "cat_units_returned"})
)
cat_return = cat_sold.merge(cat_ret, on="category", how="left")
cat_return["cat_units_returned"]   = cat_return["cat_units_returned"].fillna(0)
cat_return["return_rate_category"] = (
    cat_return["cat_units_returned"] / cat_return["cat_units_sold"]
).clip(0, 1)

# Return rate by size
size_sold = (
    order_items.merge(products[["product_id", "size"]], on="product_id")
    .groupby("size")["quantity"].sum()
    .reset_index().rename(columns={"quantity": "size_units_sold"})
)
size_ret = (
    returns.merge(products[["product_id", "size"]], on="product_id")
    .groupby("size")["return_quantity"].sum()
    .reset_index().rename(columns={"return_quantity": "size_units_returned"})
)
size_return = size_sold.merge(size_ret, on="size", how="left")
size_return["size_units_returned"] = size_return["size_units_returned"].fillna(0)
size_return["return_rate_size"]    = (
    size_return["size_units_returned"] / size_return["size_units_sold"]
).clip(0, 1)

# Return rate by return_reason (cho Q3 MCQ)
reason_dist = (
    returns.merge(products[["product_id", "category"]], on="product_id")
    .groupby(["category", "return_reason"])["return_id"]
    .count().reset_index().rename(columns={"return_id": "count"})
)

save(return_feat,    "return_features")
save(cat_return,     "return_rate_by_category")
save(size_return,    "return_rate_by_size")
save(reason_dist,    "return_reason_by_category")

ok("Return rate by category:")
for _, r in cat_return.sort_values("return_rate_category", ascending=False).iterrows():
    ok(f"  {r['category']:<20} {r['return_rate_category']*100:.2f}%")
ok("Return rate by size:")
for _, r in size_return.sort_values("return_rate_size", ascending=False).iterrows():
    ok(f"  {r['size']:<5} {r['return_rate_size']*100:.2f}%")
ok("Quadrant distribution:")
ok(str(return_feat["product_quadrant"].value_counts().to_dict()))

# ─────────────────────────────────────────────────────────
# NHÓM 6 — Daily conversion & traffic features
# Nguồn: web_traffic + orders + sales
# Dùng cho: Story 2d, Part 3 model
# ─────────────────────────────────────────────────────────

section("NHÓM 6 — Daily conversion & traffic features")

# Đơn hàng hợp lệ mỗi ngày (bỏ cancelled)
daily_orders = (
    orders[orders["order_status"].isin(["delivered", "shipped", "returned"])]
    .groupby("order_date")["order_id"].count()
    .reset_index().rename(columns={"order_date": "date", "order_id": "n_orders"})
)

# Traffic tổng hợp theo ngày
daily_traffic = (
    web_traffic.groupby("date")
    .agg(
        sessions         =("sessions",                "sum"),
        unique_visitors  =("unique_visitors",         "sum"),
        page_views       =("page_views",              "sum"),
        avg_bounce_rate  =("bounce_rate",             "mean"),
        avg_session_dur  =("avg_session_duration_sec","mean"),
    )
    .reset_index()
)

# Join với sales
daily_feat = (
    sales.merge(daily_traffic, on="date", how="left")
         .merge(daily_orders,  on="date", how="left")
)

daily_feat["n_orders"] = daily_feat["n_orders"].fillna(0)

daily_feat["conversion_rate"]     = (
    daily_feat["n_orders"] / daily_feat["sessions"].replace(0, np.nan)
)
daily_feat["revenue_per_session"] = (
    daily_feat["revenue"] / daily_feat["sessions"].replace(0, np.nan)
)
daily_feat["cogs_per_session"]    = (
    daily_feat["cogs"] / daily_feat["sessions"].replace(0, np.nan)
)

# Lag features (sessions hôm trước là leading indicator)
daily_feat = daily_feat.sort_values("date").reset_index(drop=True)
daily_feat["sessions_lag1"]  = daily_feat["sessions"].shift(1)
daily_feat["sessions_lag7"]  = daily_feat["sessions"].shift(7)
daily_feat["sessions_lag30"] = daily_feat["sessions"].shift(30)

# Rolling averages
daily_feat["sessions_ma7"]  = daily_feat["sessions"].rolling(7,  min_periods=1).mean()
daily_feat["sessions_ma30"] = daily_feat["sessions"].rolling(30, min_periods=1).mean()
daily_feat["revenue_ma7"]   = daily_feat["revenue"].rolling(7,   min_periods=1).mean()
daily_feat["revenue_ma30"]  = daily_feat["revenue"].rolling(30,  min_periods=1).mean()

# Revenue lag (dùng cho model)
daily_feat["revenue_lag1"]  = daily_feat["revenue"].shift(1)
daily_feat["revenue_lag7"]  = daily_feat["revenue"].shift(7)
daily_feat["revenue_lag365"] = daily_feat["revenue"].shift(365)   # same day last year

# YoY growth (chỉ tính được từ năm 2 trở đi)
daily_feat["revenue_yoy"] = (
    daily_feat["revenue"] / daily_feat["revenue_lag365"].replace(0, np.nan) - 1
)

# Traffic per source (cho Story 2d)
traffic_by_source = (
    web_traffic
    .merge(daily_orders.rename(columns={"date": "date"}), on="date", how="left")
    .merge(sales[["date", "revenue"]], on="date", how="left")
)
traffic_by_source["n_orders"]             = traffic_by_source["n_orders"].fillna(0)
traffic_by_source["conversion_rate"]      = (
    traffic_by_source["n_orders"] / traffic_by_source["sessions"].replace(0, np.nan)
)
traffic_by_source["revenue_per_session"]  = (
    traffic_by_source["revenue"] / traffic_by_source["sessions"].replace(0, np.nan)
)

save(daily_feat,        "daily_features")
save(traffic_by_source, "traffic_by_source")

corr = daily_feat["sessions_lag1"].corr(daily_feat["revenue"])
ok(f"avg conversion_rate:     {daily_feat['conversion_rate'].mean():.4f}")
ok(f"avg revenue_per_session: {daily_feat['revenue_per_session'].mean():.2f}")
ok(f"corr(sessions_lag1, revenue): {corr:.4f}")

# ─────────────────────────────────────────────────────────
# NHÓM 7 — Inventory loss features
# Nguồn: inventory + sales
# Dùng cho: Story 4a, Story 5.2
# ─────────────────────────────────────────────────────────

section("NHÓM 7 — Inventory loss features")

inv = inventory.copy()

# Avg daily revenue per month
monthly_revenue = (
    sales.assign(year=sales["date"].dt.year, month=sales["date"].dt.month)
    .groupby(["year", "month"])
    .agg(total_revenue=("revenue", "sum"), n_days=("revenue", "count"))
    .reset_index()
)
monthly_revenue["avg_daily_revenue"] = (
    monthly_revenue["total_revenue"] / monthly_revenue["n_days"]
)

inv_feat = inv.merge(
    monthly_revenue[["year", "month", "avg_daily_revenue"]],
    on=["year", "month"], how="left"
)

# Est. lost revenue từ stockout
inv_feat["est_lost_revenue"] = inv_feat["stockout_days"] * inv_feat["avg_daily_revenue"]

# Overstock ratio: stock_on_hand / units_sold
inv_feat["overstock_ratio"] = (
    inv_feat["stock_on_hand"] / inv_feat["units_sold"].replace(0, np.nan)
)

# ── GIẢI THÍCH DATA STRUCTURE ──────────────────────────────
# stock_on_hand = snapshot CUỐI tháng (sau khi nhập + bán)
# → KHÔNG thể dùng để detect "thiếu hàng" vì stockout xảy ra TRONG tháng
# → Thay vào đó dùng các metrics tính được từ stockout_days (đã có trong data)

# 1. Avg daily demand (units/ngày) từ lịch sử bán
inv_feat["avg_daily_demand"] = inv_feat["units_sold"] / 30

# 2. Stockout severity = % ngày trong tháng bị hết hàng
inv_feat["stockout_severity"] = (inv_feat["stockout_days"] / 30).clip(0, 1)

# 3. Fill rate đã có sẵn trong data (tỉ lệ đơn được đáp ứng đủ)
# fill_rate < 0.9 → nghiêm trọng, < 0.7 → rất nghiêm trọng
inv_feat["fill_rate_critical_flag"] = (inv_feat["fill_rate"] < 0.9).astype(int)
inv_feat["fill_rate_severe_flag"]   = (inv_feat["fill_rate"] < 0.7).astype(int)

# 4. Reorder urgency score: kết hợp stockout_days + fill_rate + days_of_supply
#    days_of_supply < 14 → cần reorder trong ~2 tuần
inv_feat["low_supply_flag"] = (inv_feat["days_of_supply"] < 14).astype(int)

# 5. Optimal reorder quantity (prescriptive — dùng cho Story 4a)
#    = demand dự kiến tháng sau + safety stock - stock hiện có
LEAD_TIME_DAYS    = 7
SAFETY_STOCK_DAYS = 7
inv_feat["optimal_reorder_qty"] = (
    inv_feat["avg_daily_demand"] * (30 + LEAD_TIME_DAYS + SAFETY_STOCK_DAYS)
    - inv_feat["stock_on_hand"]
).clip(lower=0)  # không reorder nếu stock đã dư

# 6. Waste ratio: hàng tồn quá nhiều so với nhu cầu (overstock)
inv_feat["waste_ratio"] = (
    (inv_feat["stock_on_hand"] - inv_feat["units_sold"])
    / inv_feat["stock_on_hand"].replace(0, np.nan)
).clip(0, 1)

# Q4 lost revenue summary
q4_lost = inv_feat[inv_feat["month"].isin([10, 11, 12])]["est_lost_revenue"].sum()

save(inv_feat, "inventory_features")
ok(f"est_lost_revenue total:      {inv_feat['est_lost_revenue'].sum():>20,.0f}")
ok(f"est_lost_revenue Q4 total:   {q4_lost:>20,.0f}")
ok(f"stockout_flag rate:          {inv_feat['stockout_flag'].mean()*100:.1f}% product-month")
ok(f"fill_rate_critical (<0.9):   {inv_feat['fill_rate_critical_flag'].mean()*100:.1f}%")
ok(f"fill_rate_severe   (<0.7):   {inv_feat['fill_rate_severe_flag'].mean()*100:.1f}%")
ok(f"low_supply_flag (<14d):      {inv_feat['low_supply_flag'].mean()*100:.1f}%")
ok(f"overstock_flag rate:         {inv_feat['overstock_flag'].mean()*100:.1f}%")
ok(f"avg waste_ratio:             {inv_feat['waste_ratio'].mean():.3f}")
ok(f"avg stockout_severity:       {inv_feat['stockout_severity'].mean():.3f}  (% ngày hết hàng/tháng)")

# ─────────────────────────────────────────────────────────
# NHÓM 8 — Promo active flag per day
# Nguồn: promotions + sales
# Dùng cho: Story 3, model (Part 3)
# ─────────────────────────────────────────────────────────

section("NHÓM 8 — Promo active flag per day")

promo = promotions[["promo_id", "start_date", "end_date",
                     "discount_value", "promo_type", "applicable_category"]].copy()

rows = []
for d in sales["date"]:
    active = promo[(promo["start_date"] <= d) & (promo["end_date"] >= d)]
    rows.append({
        "date"               : d,
        "n_promo_active"     : len(active),
        "promo_active_flag"  : int(len(active) > 0),
        "avg_discount_active": active["discount_value"].mean() if len(active) > 0 else 0.0,
        "max_discount_active": active["discount_value"].max()  if len(active) > 0 else 0.0,
        "n_category_promos"  : active["applicable_category"].notna().sum(),
        "n_sitewide_promos"  : active["applicable_category"].isna().sum(),
    })

promo_daily = pd.DataFrame(rows)

save(promo_daily, "promo_daily_flags")
ok(f"ngày có promo: {promo_daily['promo_active_flag'].sum():,} / {len(promo_daily):,} "
   f"({promo_daily['promo_active_flag'].mean()*100:.1f}%)")
ok(f"avg n_promo_active khi có promo: {promo_daily[promo_daily['promo_active_flag']==1]['n_promo_active'].mean():.2f}")

# ─────────────────────────────────────────────────────────
# NHÓM 9 — Channel quality (Story 2a)
# Nguồn: customer_features + cust_orders + oi_feat
# ─────────────────────────────────────────────────────────

section("NHÓM 9 — Channel quality (Story 2a)")

# Revenue, margin per customer
cust_rev_margin = cust_feat[["customer_id", "acquisition_channel",
                               "total_revenue", "total_margin",
                               "order_count", "median_inter_order_gap",
                               "retained_6m", "customer_tenure_days"]].copy()

# Return rate per customer
cust_returns = (
    returns.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
    .groupby("customer_id")["return_quantity"].sum()
    .reset_index().rename(columns={"return_quantity": "total_returned_qty"})
)
cust_items_sold = (
    order_items.merge(orders[["order_id", "customer_id"]], on="order_id", how="left")
    .groupby("customer_id")["quantity"].sum()
    .reset_index().rename(columns={"quantity": "total_qty_bought"})
)
cust_ret_rate = cust_returns.merge(cust_items_sold, on="customer_id", how="right")
cust_ret_rate["total_returned_qty"] = cust_ret_rate["total_returned_qty"].fillna(0)
cust_ret_rate["customer_return_rate"] = (
    cust_ret_rate["total_returned_qty"] / cust_ret_rate["total_qty_bought"].replace(0, np.nan)
).clip(0, 1)

cust_rev_margin = cust_rev_margin.merge(
    cust_ret_rate[["customer_id", "customer_return_rate"]], on="customer_id", how="left"
)

# Aggregate lên acquisition_channel
channel_quality = (
    cust_rev_margin.groupby("acquisition_channel")
    .agg(
        customers_count         =("customer_id",             "count"),
        revenue_per_customer    =("total_revenue",           "mean"),
        margin_per_customer     =("total_margin",            "mean"),
        retention_rate_6m       =("retained_6m",             "mean"),
        avg_inter_order_gap     =("median_inter_order_gap",  "mean"),
        avg_return_rate         =("customer_return_rate",    "mean"),
        avg_order_count         =("order_count",             "mean"),
        avg_tenure_days         =("customer_tenure_days",    "mean"),
    )
    .reset_index()
)

# CLV proxy = revenue_per_customer (đơn giản, không có CAC trong data)
channel_quality["clv_proxy"] = channel_quality["revenue_per_customer"]

# Active months proxy từ tenure
channel_quality["avg_active_months"] = channel_quality["avg_tenure_days"] / 30

save(channel_quality, "channel_quality")
ok("Channel quality summary:")
print(channel_quality[["acquisition_channel", "customers_count",
                         "revenue_per_customer", "retention_rate_6m",
                         "avg_return_rate"]].to_string(index=False))

# ─────────────────────────────────────────────────────────
# NHÓM 10 — New vs Returning revenue per year (Story 2c)
# Nguồn: orders + order_items
# Logic: đơn đầu tiên của mỗi khách = "new", còn lại = "returning"
# ─────────────────────────────────────────────────────────

section("NHÓM 10 — New vs Returning revenue per year (Story 2c)")

# Tag mỗi order: first_order hay repeat
first_order_id = (
    cust_orders.sort_values("order_date")
    .groupby("customer_id")["order_id"].first()
    .reset_index().rename(columns={"order_id": "first_order_id"})
)

# Revenue per order
order_rev = (
    oi_feat.groupby("order_id")["revenue_line"].sum()
    .reset_index().rename(columns={"revenue_line": "order_revenue"})
)

orders_tagged = (
    orders[["order_id", "customer_id", "order_date"]]
    .merge(first_order_id, on="customer_id", how="left")
    .merge(order_rev, on="order_id", how="left")
)
orders_tagged["is_first_order"]       = (
    orders_tagged["order_id"] == orders_tagged["first_order_id"]
).astype(int)
orders_tagged["year"]                 = orders_tagged["order_date"].dt.year
orders_tagged["first_order_year"]     = orders_tagged.merge(
    order_range[["customer_id", "first_order_date"]], on="customer_id"
)["first_order_date"].dt.year.values

# "new" = khách mua lần đầu TRONG năm đó (acquired_year == year)
# "returning" = khách đã mua từ năm trước trở về trước
orders_tagged["customer_type"] = np.where(
    orders_tagged["first_order_year"] == orders_tagged["year"],
    "new", "returning"
)

new_vs_returning = (
    orders_tagged.groupby(["year", "customer_type"])["order_revenue"]
    .sum().reset_index().rename(columns={"order_revenue": "revenue"})
)

# Pivot để dễ dùng
nvr_pivot = new_vs_returning.pivot(
    index="year", columns="customer_type", values="revenue"
).fillna(0).reset_index()
nvr_pivot.columns.name = None
nvr_pivot["total_revenue"]    = nvr_pivot.get("new", 0) + nvr_pivot.get("returning", 0)
nvr_pivot["pct_new"]          = nvr_pivot.get("new", 0)       / nvr_pivot["total_revenue"]
nvr_pivot["pct_returning"]    = nvr_pivot.get("returning", 0) / nvr_pivot["total_revenue"]

save(new_vs_returning, "new_vs_returning")
save(nvr_pivot,        "new_vs_returning_pivot")
ok("New vs Returning % by year:")
print(nvr_pivot[["year", "pct_new", "pct_returning"]].to_string(index=False))

# ─────────────────────────────────────────────────────────
# NHÓM 11 — Campaign before/during/after windows (Story 3b)
# Nguồn: promotions + sales
# Window: 14 ngày trước, trong campaign, 14 ngày sau
# ─────────────────────────────────────────────────────────

section("NHÓM 11 — Campaign before/during/after windows (Story 3b)")

WINDOW_DAYS = 14

# Tổng discount per campaign (proxy campaign size)
promo_size = (
    order_items[order_items["discount_amount"] > 0]
    .groupby("promo_id")["discount_amount"].sum()
    .reset_index().rename(columns={"discount_amount": "total_discount_amount"})
)
top_campaigns = promo_size.nlargest(5, "total_discount_amount")

campaign_windows = []
for _, camp in top_campaigns.iterrows():
    pid = camp["promo_id"]
    row = promotions[promotions["promo_id"] == pid]
    if row.empty:
        continue
    start = row["start_date"].iloc[0]
    end   = row["end_date"].iloc[0]

    pre_start  = start - pd.Timedelta(days=WINDOW_DAYS)
    post_end   = end   + pd.Timedelta(days=WINDOW_DAYS)

    mask_before = (sales["date"] >= pre_start) & (sales["date"] < start)
    mask_during = (sales["date"] >= start)     & (sales["date"] <= end)
    mask_after  = (sales["date"] > end)        & (sales["date"] <= post_end)

    def avg_daily(mask):
        sub = sales[mask]
        return sub["revenue"].mean() if len(sub) > 0 else np.nan

    before_avg = avg_daily(mask_before)
    during_avg = avg_daily(mask_during)
    after_avg  = avg_daily(mask_after)

    campaign_days = (end - start).days + 1
    total_days    = WINDOW_DAYS + campaign_days + WINDOW_DAYS
    baseline      = before_avg if before_avg and before_avg > 0 else np.nan

    rev_during = sales[mask_during]["revenue"].sum()
    rev_after  = sales[mask_after]["revenue"].sum()
    baseline_total = (baseline * total_days) if baseline else np.nan

    campaign_windows.append({
        "promo_id"             : pid,
        "promo_name"           : row["promo_name"].iloc[0],
        "start_date"           : start,
        "end_date"             : end,
        "campaign_days"        : campaign_days,
        "total_discount_amount": camp["total_discount_amount"],
        "avg_daily_before"     : before_avg,
        "avg_daily_during"     : during_avg,
        "avg_daily_after"      : after_avg,
        "pull_forward_ratio"   : during_avg / baseline if baseline else np.nan,
        "hangover_ratio"       : after_avg  / baseline if baseline else np.nan,
        "net_effect"           : (rev_during + rev_after - baseline_total) if baseline_total else np.nan,
    })

campaign_windows_df = pd.DataFrame(campaign_windows)

save(campaign_windows_df, "campaign_windows")
ok("Top 5 campaigns — cannibalization analysis:")
print(campaign_windows_df[["promo_name", "pull_forward_ratio",
                             "hangover_ratio", "net_effect"]].to_string(index=False))

# ─────────────────────────────────────────────────────────
# NHÓM 12 — Seasonal monthly index (Story 1b)
# monthly_index = revenue tháng x năm y / avg monthly revenue năm y
# ─────────────────────────────────────────────────────────

section("NHÓM 12 — Seasonal monthly index (Story 1b)")

monthly_sales = (
    sales.assign(year=sales["date"].dt.year, month=sales["date"].dt.month)
    .groupby(["year", "month"])["revenue"].sum()
    .reset_index()
)

annual_avg = (
    monthly_sales.groupby("year")["revenue"].mean()
    .reset_index().rename(columns={"revenue": "annual_avg_monthly_revenue"})
)

seasonal_index = monthly_sales.merge(annual_avg, on="year")
seasonal_index["monthly_index"] = (
    seasonal_index["revenue"] / seasonal_index["annual_avg_monthly_revenue"]
)

# Avg monthly index across all years (stable seasonal pattern)
avg_seasonal = (
    seasonal_index.groupby("month")["monthly_index"]
    .agg(avg_index="mean", std_index="std", cv_index=lambda x: x.std()/x.mean())
    .reset_index()
)

save(seasonal_index, "seasonal_index")
save(avg_seasonal,   "avg_seasonal_pattern")
ok("Average seasonal pattern (monthly_index):")
print(avg_seasonal.to_string(index=False))

# ─────────────────────────────────────────────────────────
# NHÓM 13 — Promo cohort vs Organic cohort (Story 3d)
# Nguồn: orders + promotions + customer cohort
# ─────────────────────────────────────────────────────────

section("NHÓM 13 — Promo cohort vs Organic cohort (Story 3d)")

# Tag mỗi khách mới: có đơn đầu trong campaign hay không
promo_periods = promotions[["start_date", "end_date"]].copy()

def is_in_campaign(dt):
    """Trả về 1 nếu ngày dt nằm trong bất kỳ campaign nào."""
    for _, r in promo_periods.iterrows():
        if r["start_date"] <= dt <= r["end_date"]:
            return 1
    return 0

# First order date per customer
first_ord = order_range[["customer_id", "first_order_date"]].copy()
# Vectorized: check ngày first_order có trong promo window không
# Dùng promo_daily_flags để vectorize nhanh hơn
first_ord = first_ord.merge(
    promo_daily[["date", "promo_active_flag"]].rename(columns={"date": "first_order_date"}),
    on="first_order_date", how="left"
)
first_ord["promo_active_flag"] = first_ord["promo_active_flag"].fillna(0).astype(int)
first_ord["cohort_type"] = first_ord["promo_active_flag"].map(
    {1: "promo_cohort", 0: "organic_cohort"}
)

# Join vào cohort retention để so sánh
cohort_retention_tagged = cohort_retention.copy()
cohort_retention_tagged["cohort_month_dt"] = cohort_retention_tagged["cohort_month"].dt.to_timestamp()

# Lấy cohort_type per customer, rồi mode per cohort_month
cust_cohort_type = (
    first_ord.assign(cohort_month=lambda x: x["first_order_date"].dt.to_period("M"))
    .groupby("cohort_month")["cohort_type"]
    .agg(lambda x: x.mode()[0])
    .reset_index()
)
cohort_retention_tagged = cohort_retention_tagged.merge(
    cust_cohort_type, on="cohort_month", how="left"
)

# Retention by cohort type × months_since_first
promo_vs_organic = (
    cohort_retention_tagged.groupby(["cohort_type", "months_since_first"])
    ["retention_rate"].mean().reset_index()
)

save(promo_vs_organic, "promo_vs_organic_retention")
ok("Promo vs Organic retention (month 1, 3, 6):")
for m in [1, 3, 6]:
    sub = promo_vs_organic[promo_vs_organic["months_since_first"] == m]
    if not sub.empty:
        ok(f"  month {m}: " + "  |  ".join(
            f"{r['cohort_type']}={r['retention_rate']:.3f}"
            for _, r in sub.iterrows()
        ))

# ─────────────────────────────────────────────────────────
# NHÓM 14 — Master daily feature table (Part 3 model)
# Gộp tất cả daily-level features vào 1 bảng
# ─────────────────────────────────────────────────────────

section("NHÓM 14 — Master daily feature table (cho model)")

master = daily_feat.copy()

# Join promo flags
master = master.merge(promo_daily, on="date", how="left")

# Join seasonal index
seasonal_lookup = seasonal_index[["year", "month", "monthly_index"]].copy()
master["year"]  = master["date"].dt.year
master["month"] = master["date"].dt.month
master = master.merge(seasonal_lookup, on=["year", "month"], how="left")

# Calendar features
master["dayofweek"]    = master["date"].dt.dayofweek
master["quarter"]      = master["date"].dt.quarter
master["is_weekend"]   = master["dayofweek"].isin([5, 6]).astype(int)
master["is_month_end"] = master["date"].dt.is_month_end.astype(int)

# Tết flag
def date_is_tet(dt):
    for start, end in tet_ranges:
        if start <= dt <= end:
            return 1
    return 0

master["is_tet_period"]  = master["date"].map(date_is_tet)
master["is_singles_day"] = (
    (master["month"] == 11) & (master["date"].dt.day == 11)
).astype(int)
master["is_double_12"]   = (
    (master["month"] == 12) & (master["date"].dt.day == 12)
).astype(int)

# YoY revenue (same day last year, already in daily_feat)
# Thêm WoW
master["revenue_lag7_wow"] = master["revenue"] / master["revenue_lag7"].replace(0, np.nan) - 1

# Drop rows không có revenue (ngày không có sales)
master = master.dropna(subset=["revenue"])

save(master, "master_daily")
ok(f"master_daily columns ({len(master.columns)}): {list(master.columns)}")
ok(f"date range: {master['date'].min().date()} → {master['date'].max().date()}")
ok(f"rows: {len(master):,}")

# ─────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────

elapsed = time.time() - t0
section("DONE")
print(f"  Thời gian  : {elapsed:.1f}s")
print(f"  Output dir : {FEAT_DIR.resolve()}")
files = sorted(FEAT_DIR.glob("*.parquet"))
print(f"  Files tạo ra: {len(files)}")
for f in files:
    df = pd.read_parquet(f)
    print(f"    {f.name:<50} {str(df.shape):<15}")

print("""
─────────────────────────────────────────────────────
  Bước tiếp theo:
  EDA   → notebooks/  (load từng file features)
  Model → src/train.py (load master_daily.parquet)
─────────────────────────────────────────────────────
""")