"""
data_prep.py
Load tất cả CSV, parse date columns, save ra data/processed/ dạng parquet.
Chạy: python src/data_prep.py
"""

import pandas as pd
from pathlib import Path
import time

RAW_DIR  = Path("data/raw")
OUT_DIR  = Path("data/processed")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
# Config: date columns theo từng file
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

# File không cần xử lý gì thêm — load thẳng
SKIP_FILES = ["sample_submission"]

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def ok(msg):   print(f"  ok   {msg}")
def warn(msg): print(f"  WARN {msg}")
def err(msg):  print(f"  ERR  {msg}")

# ─────────────────────────────────────────────
# Load, parse, save
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  data_prep.py — Load + Parse dates + Save parquet")
print("="*55)

tables = {}
t0 = time.time()

for csv_file in sorted(RAW_DIR.glob("*.csv")):
    name = csv_file.stem

    if name in SKIP_FILES:
        print(f"\n  skip {name}")
        continue

    print(f"\n  loading {name} ...")
    df = pd.read_csv(csv_file, low_memory=False)

    # parse date columns
    for col in DATE_COLS.get(name, []):
        if col not in df.columns:
            warn(f"{name}.{col}: column khong ton tai, bo qua")
            continue
        before = df[col].isna().sum()
        df[col] = pd.to_datetime(df[col], errors="coerce")
        after   = df[col].isna().sum()
        new_nulls = after - before
        if new_nulls > 0:
            warn(f"{name}.{col}: {new_nulls:,} rows khong parse duoc -> NaT")
        else:
            ok(f"{name}.{col} parsed  ({df[col].min().date()} -> {df[col].max().date()})")

    # save
    out_path = OUT_DIR / f"{name}.parquet"
    df.to_parquet(out_path, index=False)
    tables[name] = df
    ok(f"saved -> {out_path}  shape={df.shape}")

elapsed = time.time() - t0

# ─────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────

print("\n" + "="*55)
print("  DONE")
print("="*55)
print(f"  Thoi gian : {elapsed:.1f}s")
print(f"  Files saved: {len(tables)}")
print(f"  Output dir : {OUT_DIR.resolve()}")
print("""
  Tat ca file da san sang trong data/processed/
  Buoc tiep theo: python src/feature_eng.py
""")