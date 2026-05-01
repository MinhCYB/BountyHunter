# PART II — BỘ QUY TẮC LẬP TRÌNH BẮT BUỘC (SKILL.md)

Tài liệu này là luật bắt buộc. Mọi Agent phải đọc và tuân thủ toàn bộ trước khi viết bất kỳ dòng code nào.

---

## CHECKLIST VI PHẠM (Pre-Submit)

Trước khi nộp code, Agent phải tự kiểm tra toàn bộ checklist dưới đây. **Một vi phạm = từ chối.**

```
[ ] KHÔNG có print() ở bất kỳ đâu trong code
[ ] KHÔNG có for-loop trên DataFrame (df.iterrows, df.itertuples, df.apply(axis=1))
[ ] KHÔNG có integer index slicing trong CV (df.iloc[train_idx] thay vì date mask)
[ ] KHÔNG có rolling/lag feature nào thiếu .shift(1)
[ ] KHÔNG có cột target (revenue, cogs, margin) bị downcast
[ ] KHÔNG đọc config bằng hardcode — mọi tham số phải từ config.yaml
[ ] KHÔNG thiếu Type Hinting trên function signature
[ ] KHÔNG thiếu Docstring (Tiếng Việt) trên mọi function
[ ] KHÔNG thiếu random seed khi dùng model ML
[ ] KHÔNG có tqdm trên vòng lặp nhỏ (< 1000 items) — chỉ dùng khi cần
[ ] Các cột NULLABLE_BY_DESIGN giữ nguyên NaN, không impute
[ ] reorder_flag đã bị drop (CONSTANT column)
[ ] promo_id_2 được giữ nhưng không dùng làm feature (HIGH NULL)
```

---

## QUY TẮC 1: Cấm `print()` — Dùng `logging`

### SAI ❌
```python
print(f"Loaded {len(df)} rows")
print("Training complete")
```

### ĐÚNG ✅
```python
import logging

logger = logging.getLogger(__name__)

logger.info("Đã load %d dòng", len(df))
logger.info("Huấn luyện hoàn tất")
```

**Cấu hình logging tập trung trong `main.py`:**
```python
import logging
import yaml
from pathlib import Path

def setup_logging(config: dict) -> None:
    """Cấu hình logging toàn bộ pipeline từ config."""
    log_dir = Path(config["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    
    from datetime import datetime
    log_file = log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )
```

---

## QUY TẮC 2: Cấm `for-loop` trên DataFrame — Dùng Vectorization

### SAI ❌
```python
# Tính revenue_net bằng vòng lặp
results = []
for idx, row in df.iterrows():
    results.append(row["unit_price"] * row["quantity"] - row["discount_amount"])
df["revenue_net"] = results
```

```python
# Apply với axis=1 — cũng là for-loop ẩn
df["revenue_net"] = df.apply(
    lambda row: row["unit_price"] * row["quantity"] - row["discount_amount"],
    axis=1
)
```

### ĐÚNG ✅
```python
# Vectorized operation
df["revenue_net"] = df["unit_price"] * df["quantity"] - df["discount_amount"]
```

```python
# Groupby aggregate — vectorized
daily_revenue = (
    df
    .groupby("order_date")["revenue_net"]
    .sum()
    .reset_index()
    .rename(columns={"order_date": "date", "revenue_net": "daily_revenue"})
)
```

---

## QUY TẮC 3: Chống Data Leakage — BẮT BUỘC `.shift(1)`

**Nguyên tắc:** Mọi feature tính từ chuỗi thời gian (lag, rolling, MACD, elasticity) **phải được shift 1 bước** trước khi đưa vào model. Feature tại ngày `t` chỉ được dùng thông tin từ ngày `t-1` trở về trước.

### SAI ❌ — Leakage: dùng thông tin ngày hiện tại
```python
# BUG: rolling_mean tại ngày t bao gồm revenue của ngày t
df["revenue_roll7_mean"] = df["revenue"].rolling(7).mean()

# BUG: lag_1 chưa shift — đây là revenue của ngày t, không phải t-1
df["revenue_lag1"] = df["revenue"].rolling(1).mean()
```

### ĐÚNG ✅ — Shift trước khi rolling
```python
# ĐÚNG: shift(1) trước, đảm bảo window chỉ nhìn về quá khứ
df["revenue_lag1"] = df["revenue"].shift(1)
df["revenue_roll7_mean"] = df["revenue"].shift(1).rolling(7).mean()
df["revenue_roll30_std"] = df["revenue"].shift(1).rolling(30).std()
```

### ĐÚNG ✅ — MACD trên lag revenue
```python
def compute_macd(
    series: pd.Series,
    fast: int,
    slow: int,
    signal: int,
) -> pd.DataFrame:
    """
    Tính MACD trên chuỗi revenue đã shift.
    Đầu vào phải là series đã shift(1) rồi.
    """
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    macd_signal = macd_line.ewm(span=signal, adjust=False).mean()
    macd_hist = macd_line - macd_signal
    return pd.DataFrame({
        "macd_line": macd_line,
        "macd_signal": macd_signal,
        "macd_histogram": macd_hist,
    })

# Gọi đúng cách: shift(1) trước khi truyền vào
revenue_shifted = df["revenue"].shift(1)
macd_df = compute_macd(revenue_shifted, fast=12, slow=26, signal=9)
df = pd.concat([df, macd_df], axis=1)
```

---

## QUY TẮC 4: Date Masking CV — Cấm Integer Index

**Nguyên tắc:** Expanding Window CV phải lọc DataFrame bằng điều kiện trên cột date. Cấm dùng `iloc` với integer index vì khi test set ẩn có ngày bị thiếu, index sẽ sai.

### SAI ❌ — Dùng integer index
```python
from sklearn.model_selection import TimeSeriesSplit

tscv = TimeSeriesSplit(n_splits=5)
for train_idx, val_idx in tscv.split(df):
    df_train = df.iloc[train_idx]   # SAI: integer index bị lệch khi mất dòng
    df_val   = df.iloc[val_idx]
```

### ĐÚNG ✅ — Date Masking

```python
import pandas as pd
import numpy as np
from typing import Iterator, Tuple

def expanding_window_cv(
    df: pd.DataFrame,
    date_col: str,
    n_splits: int,
    min_train_days: int,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Tạo các fold Expanding Window CV bằng Date Masking.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đã sắp xếp theo date_col tăng dần.
    date_col : str
        Tên cột date (dtype datetime64).
    n_splits : int
        Số fold CV.
    min_train_days : int
        Số ngày tối thiểu trong tập train của fold đầu tiên.
    
    Yields
    ------
    (df_train, df_val) : Tuple[pd.DataFrame, pd.DataFrame]
        Mỗi fold là pair (train, validation).
    """
    all_dates = df[date_col].sort_values().unique()
    total_days = len(all_dates)
    
    val_size = (total_days - min_train_days) // n_splits
    
    for fold in range(n_splits):
        train_end_idx = min_train_days + fold * val_size
        val_end_idx   = train_end_idx + val_size
        
        train_cutoff = all_dates[train_end_idx - 1]
        val_start    = all_dates[train_end_idx]
        val_end      = all_dates[min(val_end_idx - 1, total_days - 1)]
        
        # DATE MASKING — lọc bằng điều kiện ngày, không dùng iloc
        df_train = df[df[date_col] <= train_cutoff].copy()
        df_val   = df[(df[date_col] >= val_start) & (df[date_col] <= val_end)].copy()
        
        yield df_train, df_val
```

---

## QUY TẮC 5: Config-Driven — Cấm Hardcode Tham Số

### SAI ❌
```python
# Hardcode tham số
model = LGBMRegressor(n_estimators=1000, learning_rate=0.05)
lag_days = [1, 7, 14, 30]
```

### ĐÚNG ✅
```python
import yaml
from pathlib import Path

def load_config(config_path: str = "config.yaml") -> dict:
    """Đọc toàn bộ cấu hình từ config.yaml."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

# Dùng trong code
cfg = load_config()
lag_days: list[int] = cfg["features"]["base_lag_features"]["lag_days"]
model_params: dict = cfg["models"]["lightgbm"]
model = LGBMRegressor(**model_params, random_state=cfg["models"]["random_seed"])
```

---

## QUY TẮC 6: Type Hinting và Docstring

**Mọi function phải có:** type hint đầy đủ trên parameters và return type, docstring Tiếng Việt tối thiểu mô tả mục đích + parameters + returns.

### ĐÚNG ✅
```python
import pandas as pd
from pathlib import Path

def load_sales(raw_dir: Path, date_col: str = "Date") -> pd.DataFrame:
    """
    Đọc và chuẩn hóa file sales.csv.
    
    Parameters
    ----------
    raw_dir : Path
        Thư mục chứa file CSV thô.
    date_col : str
        Tên cột ngày trong file gốc.
    
    Returns
    -------
    pd.DataFrame
        DataFrame với cột date (datetime64), revenue, cogs, margin.
        Không downcast các cột target.
    """
    path = raw_dir / "sales.csv"
    df = pd.read_csv(path, parse_dates=[date_col])
    df = df.rename(columns={date_col: "date", "Revenue": "revenue", "COGS": "cogs"})
    df["margin"] = df["cogs"] / df["revenue"]
    # KHÔNG downcast revenue, cogs, margin
    return df.sort_values("date").reset_index(drop=True)
```

---

## QUY TẮC 7: Downcast RAM — Ngoại trừ Target

```python
import pandas as pd

def downcast_df(df: pd.DataFrame, exclude_cols: list[str]) -> pd.DataFrame:
    """
    Downcast dtype để tiết kiệm RAM.
    Các cột trong exclude_cols giữ nguyên độ chính xác.
    
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
            continue  # Bỏ qua target
        
        if df[col].dtype == "float64":
            df[col] = pd.to_numeric(df[col], downcast="float")
        elif df[col].dtype == "int64":
            df[col] = pd.to_numeric(df[col], downcast="integer")
    
    return df

# Sử dụng
TARGET_COLS = ["revenue", "cogs", "margin"]
df = downcast_df(df, exclude_cols=TARGET_COLS)
```

---

## QUY TẮC 8: `tqdm` — Chỉ Dùng Cho Vòng Lặp Lớn

```python
from tqdm import tqdm

# ĐÚNG: dùng cho CV loop (n_splits fold)
for df_train, df_val in tqdm(cv_folds, desc="CV Folds", total=n_splits):
    ...

# ĐÚNG: dùng khi xử lý list file lớn
for file in tqdm(file_list, desc="Loading files"):
    ...

# SAI: không dùng cho vòng lặp < 10 items không cần theo dõi tiến độ
for col in tqdm(["a", "b", "c"]):  # Không cần tqdm ở đây
    ...
```

---

## QUY TẮC 9: `argparse` — Không dùng Makefile

```python
import argparse
from pathlib import Path

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Pipeline dự báo doanh thu")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Đường dẫn tới file config.yaml",
    )
    parser.add_argument(
        "--step",
        choices=["audit", "prep", "feature", "train", "inference", "all"],
        default="all",
        help="Bước pipeline cần chạy",
    )
    return parser.parse_args()
```

**Cách chạy:**
```bash
python main.py --config config.yaml --step all
python main.py --step train
python main.py --step inference
```

---

## QUY TẮC 10: Xử Lý NULLABLE_BY_DESIGN

```python
# ĐÚNG: Không impute các cột nullable_by_design
NULLABLE_BY_DESIGN = cfg["data"]["nullable_by_design"]

for col in NULLABLE_BY_DESIGN:
    if col in df.columns:
        pass  # Giữ nguyên NaN — không fill, không drop

# Khi tạo feature từ promo_id: dùng fillna chỉ cho cột dẫn xuất
df["has_promo"] = df["promo_id"].notna().astype("int8")
# Giữ promo_id gốc là NaN
```

---

## QUY TẮC 11: Loại Bỏ CONSTANT Column

```python
# Drop reorder_flag (CONSTANT — luôn = 0)
DROP_CONSTANT = cfg["data"]["drop_constant_cols"]  # ["reorder_flag"]

df = df.drop(columns=[c for c in DROP_CONSTANT if c in df.columns])
```

---

## Tóm Tắt Nhanh (Quick Reference)

| Vi phạm | Thay thế |
|---|---|
| `print()` | `logger.info()` / `logger.warning()` |
| `df.iterrows()` | Vectorized pandas operation |
| `df.apply(axis=1)` | Vectorized pandas operation |
| `df.iloc[idx]` trong CV | `df[df[date_col] <= cutoff]` |
| `df["col"].rolling(n).mean()` | `df["col"].shift(1).rolling(n).mean()` |
| Hardcode tham số | `cfg["section"]["key"]` từ config.yaml |
| `float64` / `int64` không cần thiết | `pd.to_numeric(..., downcast=...)` |
| Function không có type hint | `def fn(x: pd.DataFrame) -> pd.Series:` |
| Function không có docstring | Thêm docstring Tiếng Việt |
