"""
inference.py — Agent 3: Dự báo Revenue và Margin cho tập test, xuất submission.csv.

Pipeline:
  1. Load model_revenue.pkl và model_margin.pkl từ thư mục model mới nhất
  2. Load feature_table.parquet, lọc tập test bằng Date Masking
  3. Predict revenue và margin
  4. Tính COGS = Revenue × Margin
  5. Sắp xếp theo thứ tự của sample_submission.csv
  6. Validate và xuất outputs/submission.csv
"""

from __future__ import annotations

import argparse
import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import numpy as np
import yaml

import __main__
try:
    from train import HybridRegressor
except ImportError:
    from src.train import HybridRegressor
__main__.HybridRegressor = HybridRegressor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments cho inference.py."""
    parser = argparse.ArgumentParser(
        description="Inference: dự báo Revenue và Margin cho tập test"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Đường dẫn tới file config.yaml",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=None,
        help="Thư mục chứa model (mặc định: thư mục mới nhất trong models/)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# 2. Config
# ---------------------------------------------------------------------------

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
        Dictionary chứa toàn bộ config.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# 3. Model Loading
# ---------------------------------------------------------------------------

def _find_latest_model_dir(base_model_dir: Path) -> Path:
    """
    Tìm thư mục model mới nhất (sắp xếp theo tên timestamp).

    Parameters
    ----------
    base_model_dir : Path
        Thư mục gốc chứa các thư mục model.

    Returns
    -------
    Path
        Thư mục model mới nhất.

    Raises
    ------
    FileNotFoundError
        Nếu không tìm thấy thư mục model nào.
    """
    subdirs = sorted(
        [d for d in base_model_dir.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )
    if not subdirs:
        raise FileNotFoundError(f"Không tìm thấy thư mục model trong: {base_model_dir}")
    latest = subdirs[0]
    logger.info("Sử dụng model directory: %s", latest)
    return latest


def load_model(model_dir: Path, cfg: dict = None, feature_cols: List[str] = None) -> Tuple[Any, Any]:
    """
    Load model_revenue.pkl và model_margin.pkl từ thư mục model.

    Parameters
    ----------
    model_dir : Path
        Thư mục chứa hai file pkl.
    cfg : dict, optional
        Config đầy đủ
    feature_cols : List[str], optional
        Danh sách feature cols

    Returns
    -------
    Tuple[Any, Any]
        Cặp (model_revenue, model_margin).

    Raises
    ------
    FileNotFoundError
        Nếu thiếu một trong hai file model.
    """
    path_rev = model_dir / "model_revenue.pkl"
    path_mar = model_dir / "model_margin.pkl"

    for p in (path_rev, path_mar):
        if not p.exists():
            raise FileNotFoundError(f"Không tìm thấy model file: {p}")

    with open(path_rev, "rb") as f:
        model_revenue = pickle.load(f)
    logger.info("Đã load model_revenue từ: %s", path_rev)

    with open(path_mar, "rb") as f:
        model_margin = pickle.load(f)
    logger.info("Đã load model_margin từ: %s", path_mar)

    return model_revenue, model_margin


# ---------------------------------------------------------------------------
# 4. Data Loading
# ---------------------------------------------------------------------------

def load_feature_table(processed_dir: Path) -> pd.DataFrame:
    """
    Load test_features.parquet từ thư mục processed.

    Parameters
    ----------
    processed_dir : Path
        Thư mục chứa test_features.parquet.

    Returns
    -------
    pd.DataFrame
        DataFrame đã sắp xếp theo date tăng dần.
    """
    path = processed_dir / "test_features.parquet"
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Đã load test_features: %d dòng", len(df))
    return df


def load_feature_cols(model_dir: Path) -> List[str]:
    """
    Load danh sách feature columns đã được lưu lúc train.

    Đảm bảo inference dùng đúng thứ tự và tập cột mà model đã được fit.

    Parameters
    ----------
    model_dir : Path
        Thư mục model chứa feature_cols.json.

    Returns
    -------
    List[str]
        Danh sách tên cột feature theo thứ tự train.

    Raises
    ------
    FileNotFoundError
        Nếu không tìm thấy feature_cols.json.
    """
    import json
    path = model_dir / "feature_cols.json"
    if not path.exists():
        raise FileNotFoundError(f"Không tìm thấy feature_cols.json: {path}")
    with open(path, "r", encoding="utf-8") as f:
        feature_cols = json.load(f)
    logger.info("Đã load feature_cols: %d cột", len(feature_cols))
    return feature_cols





def _postprocess_predictions(
    result: pd.DataFrame,
    df_train_stats: dict,
    cfg: dict,
) -> pd.DataFrame:
    """
    Post-process predictions: clip and winsorize.
    Đọc cấu hình từ cfg["post_processing"].
    
    Parameters
    ----------
    result : pd.DataFrame
        DataFrame kết quả với các cột Revenue và margin_pred.
    df_train_stats : dict
        Dictionary chứa thống kê từ train (revenue_p99).
    cfg : dict
        Cấu hình pipeline đầy đủ.
        
    Returns
    -------
    pd.DataFrame
        DataFrame kết quả sau khi xử lý.
    """
    pp_cfg = cfg["post_processing"]
    if not pp_cfg["enabled"]:
        logger.info("post_processing disabled — skipping")
        return result

    # 1. Clip Revenue âm
    result["Revenue"] = result["Revenue"].clip(lower=pp_cfg["revenue_clip_lower"])
    
    # 2. Clip Margin
    result["margin_pred"] = result["margin_pred"].clip(
        pp_cfg["margin_clip_lower"], pp_cfg["margin_clip_upper"]
    )
    
    # 3. Winsorize Revenue
    if pp_cfg["revenue_winsorize_enabled"]:
        multiplier = pp_cfg["revenue_winsorize_multiplier"]
        upper_bound = df_train_stats["revenue_p99"] * multiplier
        result["Revenue"] = result["Revenue"].clip(upper=upper_bound)
        
    # 4. Recompute COGS
    result["COGS"] = result["Revenue"] * result["margin_pred"]
    
    return result


def build_submission(
    df: pd.DataFrame,
    model_rev: Any,
    model_mar: Any,
    feature_cols: List[str],
    date_col: str,
    sample_sub_path: Path,
    df_train_stats: dict,
    cfg: dict,
) -> pd.DataFrame:
    """
    Dự báo Revenue và Margin, tính COGS, sắp xếp theo sample_submission.

    Parameters
    ----------
    df : pd.DataFrame
        Feature table đã lọc tập test.
    model_rev : Any
        Model dự báo Revenue.
    model_mar : Any
        Model dự báo Margin.
    feature_cols : List[str]
        Danh sách cột feature load từ feature_cols.json (đúng thứ tự train).
    date_col : str
        Tên cột date trong feature_table.
    sample_sub_path : Path
        Đường dẫn sample_submission.csv.
    df_train_stats : dict
        Dictionary chứa thống kê train (ví dụ: revenue_p99).
    cfg : dict
        Config đầy đủ.

    Returns
    -------
    pd.DataFrame
        DataFrame với 3 cột: Date, Revenue, COGS — theo thứ tự của sample_submission.
    """
    model_name: str = cfg["models"]["active"]

    logger.info("Số feature dùng cho inference: %d", len(feature_cols))

    X_test = df[feature_cols].copy()
    drop_cols = ["sample_weight", "is_covid_period"]
    X_test = X_test.drop(columns=[c for c in drop_cols if c in X_test.columns])
    
    hybrid_enabled: bool = cfg.get("hybrid", {}).get("enabled", False)

    # Predict Revenue
    if hybrid_enabled:
        # HybridRegressor.predict() needs the full df slice (uses date internally)
        # It returns predictions already in original scale — DO NOT call expm1
        rev_preds = model_rev.predict(df[feature_cols + ["date"]].copy() if "date" in df.columns else df[feature_cols].copy())
    else:
        # Pure XGBoost: output is log-scale, must invert exactly once
        rev_preds = model_rev.predict(X_test)
        rev_preds = np.expm1(rev_preds)  # ONE TIME. Never again downstream.

    rev_preds = np.clip(rev_preds, 0, 6500000)

    # Predict Margin
    if hybrid_enabled:
        mar_preds = model_mar.predict(df[feature_cols + ["date"]].copy() if "date" in df.columns else df[feature_cols].copy())
    else:
        mar_preds = model_mar.predict(X_test)

    mar_preds = np.clip(mar_preds, 0, None)

    # Khởi tạo result
    result = df[[date_col]].copy()
    result["Revenue"] = rev_preds
    result["margin_pred"] = mar_preds
    result["COGS"] = result["Revenue"] * result["margin_pred"]

    # Post-process (clip and winsorize)
    result = _postprocess_predictions(result, df_train_stats, cfg)
    result = result.rename(columns={date_col: "Date"})
    result["Date"] = pd.to_datetime(result["Date"])

    logger.info(
        "Dự báo hoàn tất: Revenue mean=%.2f | COGS mean=%.2f",
        result["Revenue"].mean(),
        result["COGS"].mean(),
    )

    # Load sample_submission để lấy thứ tự ngày chuẩn
    sample_sub = pd.read_csv(sample_sub_path, parse_dates=["Date"])
    logger.info("sample_submission: %d dòng", len(sample_sub))

    # Merge theo thứ tự sample_submission — không sort lại
    submission = sample_sub[["Date"]].merge(
        result[["Date", "Revenue", "COGS"]],
        on="Date",
        how="left",
    )

    return submission[["Date", "Revenue", "COGS"]]


# ---------------------------------------------------------------------------
# 6. Save & Validate
# ---------------------------------------------------------------------------

def save_submission(submission: pd.DataFrame, output_dir: Path) -> None:
    """
    Validate và lưu submission.csv ra thư mục output.

    Parameters
    ----------
    submission : pd.DataFrame
        DataFrame kết quả với 3 cột: Date, Revenue, COGS.
    output_dir : Path
        Thư mục đầu ra.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "submission.csv"
    submission.to_csv(out_path, index=False)
    logger.info("Đã lưu submission: %s (%d dòng)", out_path, len(submission))


def _validate_submission(
    submission: pd.DataFrame,
    sample_sub_path: Path,
) -> None:
    """
    Kiểm tra số dòng và giá trị null của submission so với sample_submission.

    Parameters
    ----------
    submission : pd.DataFrame
        DataFrame kết quả đã build.
    sample_sub_path : Path
        Đường dẫn sample_submission.csv.
    """
    sample_sub = pd.read_csv(sample_sub_path)
    expected_rows = len(sample_sub)
    actual_rows = len(submission)

    if actual_rows != expected_rows:
        logger.error(
            "Số dòng KHÔNG khớp! Expected=%d, Actual=%d",
            expected_rows,
            actual_rows,
        )
    else:
        logger.info("Validate OK: %d dòng khớp với sample_submission", actual_rows)

    # Kiểm tra null
    null_counts = submission[["Revenue", "COGS"]].isnull().sum()
    if null_counts.any():
        logger.error("Phát hiện giá trị null trong submission:\n%s", null_counts[null_counts > 0])
    else:
        logger.info("Validate OK: không có giá trị null trong Revenue và COGS")

    # Kiểm tra Revenue và COGS dương
    n_neg_rev = (submission["Revenue"] < 0).sum()
    n_neg_cogs = (submission["COGS"] < 0).sum()
    if n_neg_rev > 0:
        logger.warning("Phát hiện %d dòng Revenue âm", n_neg_rev)
    if n_neg_cogs > 0:
        logger.warning("Phát hiện %d dòng COGS âm", n_neg_cogs)


# ---------------------------------------------------------------------------
# 7. Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Điểm vào chính của inference.py.

    Luồng:
      1. Parse args, load config
      2. Load model (thư mục mới nhất hoặc chỉ định qua --model-dir)
      3. Load feature_table, lọc tập test bằng Date Masking
      4. Build submission (predict revenue, margin, tính COGS)
      5. Validate và save
    """
    args = parse_args()
    cfg = load_config(args.config)

    # Setup logging
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}_inference.log"
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"]),
        format=cfg["logging"]["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    processed_dir = Path(cfg["paths"]["processed_dir"])
    base_model_dir = Path(cfg["paths"]["model_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])
    raw_dir = Path(cfg["paths"]["raw_dir"])
    sample_sub_path = raw_dir / "sample_submission.csv"

    date_col_cfg: str = cfg["data"]["date_col"]   # "Date" (raw)
    test_start = pd.Timestamp(cfg["data"]["test_start"])
    test_end = pd.Timestamp(cfg["data"]["test_end"])

    # Resolve model directory
    if args.model_dir is not None:
        model_dir = args.model_dir
    else:
        model_dir = _find_latest_model_dir(base_model_dir)

    # Load feature_cols đúng thứ tự từ lúc train
    feature_cols = load_feature_cols(model_dir)

    # Load model
    model_revenue, model_margin = load_model(model_dir, cfg, feature_cols)

    # Load feature_table
    df = load_feature_table(processed_dir)

    # Lọc tập test bằng Date Masking (không dùng iloc)
    df["date"] = pd.to_datetime(df["date"])
    df_test = df[
        (df["date"] >= test_start) & (df["date"] <= test_end)
    ].copy()

    logger.info(
        "Tập test: %d dòng (%s → %s)",
        len(df_test),
        df_test["date"].min().date() if len(df_test) > 0 else "N/A",
        df_test["date"].max().date() if len(df_test) > 0 else "N/A",
    )

    if len(df_test) == 0:
        logger.error("Tập test rỗng — kiểm tra lại test_start và test_end trong config")
        return

    # Load train stats
    import json
    train_stats_path = processed_dir / "train_stats.json"
    if train_stats_path.exists():
        with open(train_stats_path, "r", encoding="utf-8") as f:
            df_train_stats = json.load(f)
    else:
        logger.warning("Không tìm thấy train_stats.json, dùng fallback revenue_p99 = inf")
        df_train_stats = {"revenue_p99": float("inf")}

    # Build submission
    submission = build_submission(
        df=df_test,
        model_rev=model_revenue,
        model_mar=model_margin,
        feature_cols=feature_cols,
        date_col="date",
        sample_sub_path=sample_sub_path,
        df_train_stats=df_train_stats,
        cfg=cfg,
    )

    # Validate
    _validate_submission(submission, sample_sub_path)

    # Save
    save_submission(submission, output_dir)

    logger.info("Inference hoàn tất. Submission: %s", output_dir / "submission.csv")


if __name__ == "__main__":
    main()