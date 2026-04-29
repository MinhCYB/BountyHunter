"""
evaluate_baseline.py — Đánh giá Naive Lag-365 Baseline
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from train import expanding_window_cv

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate Naive Lag-365 Baseline")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Đường dẫn tới file config.yaml",
    )
    return parser.parse_args()


def load_config(path: Path) -> dict:
    """Load config từ yaml."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(cfg: dict) -> None:
    """Thiết lập logging theo config."""
    level = getattr(logging, cfg["logging"]["level"])
    fmt = cfg["logging"]["format"]
    logging.basicConfig(
        level=level, 
        format=fmt,
        handlers=[logging.StreamHandler()]
    )


def load_feature_table(processed_dir: Path) -> pd.DataFrame:
    """Load train_features.parquet và sort theo date."""
    path = processed_dir / "train_features.parquet"
    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def predict_naive_baseline(
    df_predict: pd.DataFrame, df_history: pd.DataFrame, target_col: str
) -> pd.Series:
    """
    Dự báo bằng Naive Baseline (Lag 365).
    Fallback: 364 hoặc 366.
    
    Parameters
    ----------
    df_predict : pd.DataFrame
        DataFrame cần dự báo (chứa cột 'date').
    df_history : pd.DataFrame
        DataFrame chứa lịch sử (để tra cứu).
    target_col : str
        Tên cột mục tiêu.
        
    Returns
    -------
    pd.Series
        Series chứa giá trị dự báo, có cùng index với df_predict.
    """
    target_map = df_history.set_index("date")[target_col]
    
    dates = df_predict["date"]
    
    preds_365 = target_map.reindex(dates - pd.Timedelta(days=365)).values
    preds_364 = target_map.reindex(dates - pd.Timedelta(days=364)).values
    preds_366 = target_map.reindex(dates - pd.Timedelta(days=366)).values
    
    preds = pd.Series(preds_365, index=df_predict.index)
    preds_364 = pd.Series(preds_364, index=df_predict.index)
    preds_366 = pd.Series(preds_366, index=df_predict.index)
    
    preds = preds.fillna(preds_364).fillna(preds_366)
    return preds


def evaluate_metrics(y_true: pd.Series, y_pred: pd.Series) -> tuple[float, float, float]:
    """
    Tính MAE, RMSE, R2 cho dữ liệu không bị NaN.
    
    Parameters
    ----------
    y_true : pd.Series
        Series chứa giá trị thực.
    y_pred : pd.Series
        Series chứa giá trị dự báo.
        
    Returns
    -------
    tuple[float, float, float]
        Cặp (MAE, RMSE, R2).
    """
    mask = y_true.notna() & y_pred.notna()
    if mask.sum() == 0:
        return np.nan, np.nan, np.nan
        
    y_t = y_true[mask]
    y_p = y_pred[mask]
    
    mae = mean_absolute_error(y_t, y_p)
    rmse = np.sqrt(mean_squared_error(y_t, y_p))
    r2 = r2_score(y_t, y_p)
    return float(mae), float(rmse), float(r2)


def main() -> None:
    """Điểm vào chính của evaluate_baseline.py."""
    args = parse_args()
    cfg = load_config(args.config)
    setup_logging(cfg)

    processed_dir = Path(cfg["paths"]["processed_dir"])
    df = load_feature_table(processed_dir)
    target_revenue = cfg["data"]["target_revenue"]
    
    if target_revenue not in df.columns:
        raise ValueError(f"Không tìm thấy cột target_revenue: {target_revenue} trong dữ liệu.")
        
    # 3. Đánh giá trên CV folds
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    df_train_full = df[df["date"] <= train_end].copy()
    
    cv_folds_metrics = []
    logger.info("--- Bắt đầu đánh giá CV ---")
    for i, (df_tr, df_val) in enumerate(expanding_window_cv(df_train_full, cfg), 1):
        preds = predict_naive_baseline(df_val, df, target_revenue)
        
        valid_mask = preds.notna() & df_val[target_revenue].notna()
        n_dropped = len(df_val) - valid_mask.sum()
        if n_dropped > 0:
            logger.warning("Fold %d: Loại bỏ %d dòng do không tìm thấy lag-365", i, n_dropped)
            
        mae, rmse, r2 = evaluate_metrics(df_val[target_revenue], preds)
        
        logger.info("Fold %d Baseline — MAE=%.2f | RMSE=%.2f | R²=%.4f", i, mae, rmse, r2)
        cv_folds_metrics.append({
            "fold": i,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2
        })
        
    cv_mean = {
        "MAE": float(np.nanmean([m["MAE"] for m in cv_folds_metrics])),
        "RMSE": float(np.nanmean([m["RMSE"] for m in cv_folds_metrics])),
        "R2": float(np.nanmean([m["R2"] for m in cv_folds_metrics]))
    }
    cv_std = {
        "MAE": float(np.nanstd([m["MAE"] for m in cv_folds_metrics])),
        "RMSE": float(np.nanstd([m["RMSE"] for m in cv_folds_metrics])),
        "R2": float(np.nanstd([m["R2"] for m in cv_folds_metrics]))
    }
    logger.info("CV MEAN Baseline — MAE=%.2f ± %.2f | RMSE=%.2f ± %.2f | R²=%.4f ± %.4f",
                cv_mean["MAE"], cv_std["MAE"],
                cv_mean["RMSE"], cv_std["RMSE"],
                cv_mean["R2"], cv_std["R2"])
                
    # 4. Đánh giá trên TEST SET
    test_start = pd.Timestamp(cfg["data"]["test_start"])
    test_end = pd.Timestamp(cfg["data"]["test_end"])
    df_test = df[(df["date"] >= test_start) & (df["date"] <= test_end)].copy()
    
    test_metrics_res = {}
    if len(df_test) > 0:
        logger.info("--- Bắt đầu đánh giá Test Set ---")
        preds_test = predict_naive_baseline(df_test, df, target_revenue)
        valid_mask = preds_test.notna() & df_test[target_revenue].notna()
        n_valid = valid_mask.sum()
        n_dropped = len(df_test) - n_valid
        
        if n_dropped > 0:
            logger.warning("Test Set: Loại bỏ %d dòng do không tìm thấy lag-365", n_dropped)
            
        mae_t, rmse_t, r2_t = evaluate_metrics(df_test[target_revenue], preds_test)
        logger.info("TEST SET Baseline — MAE=%.2f | RMSE=%.2f | R²=%.4f | n_valid=%d | n_dropped=%d",
                    mae_t, rmse_t, r2_t, n_valid, n_dropped)
                    
        test_metrics_res = {
            "MAE": mae_t,
            "RMSE": rmse_t,
            "R2": r2_t,
            "n_valid": int(n_valid),
            "n_dropped": int(n_dropped)
        }
    else:
        logger.info("Không có dữ liệu Test Set trong train_features.parquet.")
        
    # 5. So sánh với model
    model_dir = Path(cfg["paths"]["model_dir"])
    model_metrics = None
    comparison = None
    
    if model_dir.exists():
        subdirs = sorted([d for d in model_dir.iterdir() if d.is_dir()], key=lambda x: x.name, reverse=True)
        for d in subdirs:
            metrics_path = d / "cv_metrics.json"
            if metrics_path.exists():
                with open(metrics_path, "r", encoding="utf-8") as f:
                    try:
                        cv_res = json.load(f)
                        if target_revenue in cv_res:
                            model_metrics = cv_res[target_revenue]["mean"]
                            break
                    except json.JSONDecodeError:
                        pass
                        
    if model_metrics is not None:
        logger.info("--- So sánh Baseline vs Model ---")
        m_mae = model_metrics["mae"]
        m_rmse = model_metrics["rmse"]
        m_r2 = model_metrics["r2"]
        
        d_mae = m_mae - cv_mean["MAE"]
        d_rmse = m_rmse - cv_mean["RMSE"]
        d_r2 = m_r2 - cv_mean["R2"]
        
        logger.info("Metric     | Baseline | Model  | Delta")
        logger.info("MAE        | %8.2f | %6.2f | %+7.2f %s", cv_mean["MAE"], m_mae, d_mae, "(+ nếu model tệ hơn)" if d_mae > 0 else "")
        logger.info("RMSE       | %8.2f | %6.2f | %+7.2f", cv_mean["RMSE"], m_rmse, d_rmse)
        logger.info("R²         | %8.4f | %6.4f | %+7.4f", cv_mean["R2"], m_r2, d_r2)
        
        comparison = {
            "MAE": {"baseline": cv_mean["MAE"], "model": m_mae, "delta": d_mae},
            "RMSE": {"baseline": cv_mean["RMSE"], "model": m_rmse, "delta": d_rmse},
            "R2": {"baseline": cv_mean["R2"], "model": m_r2, "delta": d_r2}
        }
    else:
        logger.warning("Không tìm thấy cv_metrics — bỏ qua bước so sánh")
        
    # 6. Export kết quả
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    out_file = output_dir / "baseline_metrics.json"
    
    final_output = {
        "cv_folds": cv_folds_metrics,
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "test_set": test_metrics_res,
        "comparison": comparison
    }
    
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
        
    logger.info("Đã lưu kết quả baseline vào: %s", out_file)

if __name__ == "__main__":
    main()
