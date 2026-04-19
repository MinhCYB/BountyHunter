"""
00_data_audit.py  —  Bước 1: Kiểm tra cơ bản
Chạy: python 00_data_audit.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

RAW_DIR = Path("data/raw")

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

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

FK_CHECKS = [
    # (child_table,  child_col,    parent_table,  parent_col)
    ("orders",       "customer_id", "customers",   "customer_id"),
    ("orders",       "zip",         "geography",   "zip"),
    ("order_items",  "order_id",    "orders",      "order_id"),
    ("order_items",  "product_id",  "products",    "product_id"),
    ("order_items",  "promo_id",    "promotions",  "promo_id"),
    ("payments",     "order_id",    "orders",      "order_id"),
    ("shipments",    "order_id",    "orders",      "order_id"),
    ("returns",      "order_id",    "orders",      "order_id"),
    ("returns",      "product_id",  "products",    "product_id"),
    ("reviews",      "order_id",    "orders",      "order_id"),
    ("reviews",      "customer_id", "customers",   "customer_id"),
    ("inventory",    "product_id",  "products",    "product_id"),
    ("customers",    "zip",         "geography",   "zip"),
]

# Nullable by design — không phải bug khi null
NULLABLE_BY_DESIGN = {
    "customers"  : ["gender", "age_group", "acquisition_channel"],
    "promotions" : ["applicable_category", "promo_channel", "min_order_value"],
    "order_items": ["promo_id", "promo_id_2"],
}

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def sub(title):
    print(f"\n  -- {title}")

def ok(msg):   print(f"    ok   {msg}")
def warn(msg): print(f"    WARN {msg}")
def err(msg):  print(f"    ERR  {msg}")

def pct(n, total):
    return f"{n:,} ({n/total*100:.1f}%)" if total else "0"

# ─────────────────────────────────────────────
# Load
# ─────────────────────────────────────────────

section("LOADING ALL CSV FILES")

tables = {}
for f in sorted(RAW_DIR.glob("*.csv")):
    name = f.stem
    try:
        df = pd.read_csv(f, low_memory=False)
        tables[name] = df
        print(f"    ok   {name:<20} {df.shape[0]:>8,} rows  x  {df.shape[1]} cols")
    except Exception as e:
        err(f"{name}: {e}")

# ─────────────────────────────────────────────
# CHECK 1 — NULL ANALYSIS
# ─────────────────────────────────────────────

section("CHECK 1 — NULL ANALYSIS")

for name, df in tables.items():
    sub(name)
    total    = len(df)
    nullable = NULLABLE_BY_DESIGN.get(name, [])
    found    = False

    for col in df.columns:
        n = df[col].isna().sum()
        if n == 0:
            continue
        found = True
        ratio = n / total

        if col in nullable:
            ok(f"{col:<35} null: {pct(n, total)}  (nullable by design)")
        elif ratio > 0.3:
            err(f"{col:<35} null: {pct(n, total)}  << CRITICAL")
        elif ratio > 0.05:
            warn(f"{col:<35} null: {pct(n, total)}  << can xu ly")
        else:
            warn(f"{col:<35} null: {pct(n, total)}  << nho, fill duoc")

    if not found:
        ok("Khong co null")

# ─────────────────────────────────────────────
# CHECK 2 — DATE DTYPE
# ─────────────────────────────────────────────

section("CHECK 2 — DATE COLUMNS DTYPE")

for name, cols in DATE_COLS.items():
    if name not in tables:
        warn(f"{name}: khong tim thay file")
        continue
    df = tables[name]
    sub(name)
    for col in cols:
        if col not in df.columns:
            err(f"{col}: column khong ton tai")
            continue

        dtype = str(df[col].dtype)

        if dtype == "object":
            sample = df[col].dropna().iloc[0] if df[col].notna().any() else None
            parsed = pd.to_datetime(df[col], errors="coerce")
            n_fail = parsed.isna().sum() - df[col].isna().sum()
            if n_fail == 0:
                warn(f"{col:<25} dtype=object  -> parse duoc, can them pd.to_datetime()  sample='{sample}'")
            else:
                err(f"{col:<25} dtype=object  -> {n_fail:,} rows khong parse duoc  sample='{sample}'")
        elif "datetime" in dtype:
            mn = df[col].min()
            mx = df[col].max()
            ok(f"{col:<25} dtype={dtype}  range: {mn} -> {mx}")
        else:
            warn(f"{col:<25} dtype={dtype}  << khong phai datetime, kiem tra lai")

# ─────────────────────────────────────────────
# CHECK 3 — FK INTEGRITY
# ─────────────────────────────────────────────

section("CHECK 3 — FOREIGN KEY INTEGRITY")

for child_name, child_col, parent_name, parent_col in FK_CHECKS:
    if child_name not in tables or parent_name not in tables:
        continue

    child_df  = tables[child_name]
    parent_df = tables[parent_name]

    if child_col not in child_df.columns or parent_col not in parent_df.columns:
        warn(f"{child_name}.{child_col} -> {parent_name}.{parent_col}: column missing")
        continue

    child_vals  = child_df[child_col].dropna()
    parent_vals = set(parent_df[parent_col].dropna())
    orphans     = child_vals[~child_vals.isin(parent_vals)]
    n_orphan    = len(orphans)
    total       = len(child_vals)
    label       = f"{child_name}.{child_col:<15} -> {parent_name}.{parent_col:<15}"

    if n_orphan == 0:
        ok(f"{label}  orphans=0")
    else:
        ratio = n_orphan / total
        msg   = f"{label}  orphans={pct(n_orphan, total)}"
        if ratio > 0.05:
            err(msg + "  << CRITICAL")
        else:
            warn(msg + "  << nho, log lai khi drop")

# ─────────────────────────────────────────────
# CHECK 4 — BUSINESS CONSTRAINTS
# ─────────────────────────────────────────────

section("CHECK 4 — BUSINESS CONSTRAINT & RANGE CHECKS")

# 4a. cogs < price
sub("products: cogs < price")
if "products" in tables:
    p   = tables["products"]
    bad = p[p["cogs"] >= p["price"]]
    if bad.empty:
        ok("Tat ca san pham co cogs < price")
    else:
        err(f"{len(bad):,} san pham vi pham cogs >= price")
        print(bad[["product_id", "product_name", "category", "cogs", "price"]].head(10).to_string(index=False))

# 4b. rating in [1, 5]
sub("reviews: rating trong [1, 5]")
if "reviews" in tables:
    r   = tables["reviews"]
    bad = r[~r["rating"].between(1, 5)]
    if bad.empty:
        ok("Tat ca rating nam trong [1, 5]")
    else:
        err(f"{len(bad):,} reviews co rating ngoai [1, 5]")
        print(r["rating"].value_counts().sort_index().to_string())

# 4c. quantity > 0
sub("order_items: quantity > 0")
if "order_items" in tables:
    oi  = tables["order_items"]
    bad = oi[oi["quantity"] <= 0]
    if bad.empty:
        ok("Tat ca quantity > 0")
    else:
        err(f"{len(bad):,} rows co quantity <= 0")

# 4d. refund_amount >= 0
sub("returns: refund_amount >= 0")
if "returns" in tables:
    r   = tables["returns"]
    bad = r[r["refund_amount"] < 0]
    if bad.empty:
        ok("Tat ca refund_amount >= 0")
    else:
        err(f"{len(bad):,} rows co refund_amount am")

# 4e. payment_value > 0
sub("payments: payment_value > 0")
if "payments" in tables:
    pay = tables["payments"]
    bad = pay[pay["payment_value"] <= 0]
    if bad.empty:
        ok("Tat ca payment_value > 0")
    else:
        err(f"{len(bad):,} rows co payment_value <= 0")
        print(pay["payment_value"].describe())

# 4f. ship_date <= delivery_date
sub("shipments: ship_date <= delivery_date")
if "shipments" in tables:
    s = tables["shipments"].copy()
    s["ship_date"]     = pd.to_datetime(s["ship_date"],     errors="coerce")
    s["delivery_date"] = pd.to_datetime(s["delivery_date"], errors="coerce")
    bad = s[s["ship_date"] > s["delivery_date"]]
    if bad.empty:
        ok("Tat ca delivery_date >= ship_date")
    else:
        err(f"{len(bad):,} rows co delivery_date < ship_date")

# 4g. fill_rate, sell_through_rate trong [0, 1]
sub("inventory: fill_rate va sell_through_rate trong [0, 1]")
if "inventory" in tables:
    inv = tables["inventory"]
    for col in ["fill_rate", "sell_through_rate"]:
        if col not in inv.columns:
            continue
        bad = inv[~inv[col].between(0, 1)]
        if bad.empty:
            ok(f"{col} nam trong [0, 1]")
        else:
            err(f"{col}: {len(bad):,} rows ngoai [0, 1]  min={inv[col].min():.4f}  max={inv[col].max():.4f}")

# 4h. bounce_rate trong [0, 1]
sub("web_traffic: bounce_rate trong [0, 1]")
if "web_traffic" in tables:
    wt  = tables["web_traffic"]
    bad = wt[~wt["bounce_rate"].between(0, 1)]
    if bad.empty:
        ok("Tat ca bounce_rate trong [0, 1]")
    else:
        err(f"{len(bad):,} rows ngoai [0, 1]  min={wt['bounce_rate'].min():.4f}  max={wt['bounce_rate'].max():.4f}")

# ─────────────────────────────────────────────
# TONG KET
# ─────────────────────────────────────────────

section("TONG KET — VIEC CAN LAM TRONG data_prep.py")
print("""
  NULL:
  [ ] Cot nao null ngoai nullable_by_design -> quyet dinh DROP hay FILL
  [ ] Neu null < 5%  -> fill bang median (numeric) hoac mode (categorical)
  [ ] Neu null > 30% -> can nhac drop cot hoac flag thanh "Unknown"

  DATE:
  [ ] Cot nao dtype=object -> them pd.to_datetime() trong data_prep
  [ ] Cot nao parse fail   -> xem sample, xac dinh format thu cong

  FK:
  [ ] Orphan rows   -> log ra file, sau do DROP truoc khi join
  [ ] Orphan > 5%   -> dieu tra nguyen nhan truoc khi drop

  CONSTRAINT:
  [ ] cogs >= price      -> flag lai, khong drop
  [ ] rating ngoai [1,5] -> DROP row
  [ ] quantity <= 0      -> DROP row
  [ ] ship > deliver     -> flag, khong drop (co the timezone issue)
""")