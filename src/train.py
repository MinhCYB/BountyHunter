"""
train.py — Agent 3: Huấn luyện mô hình dự báo Revenue và Margin.

Pipeline:
  1. Load feature_table.parquet
  2. Expanding Window CV với Date Masking
  3. Train final model trên toàn bộ tập train
  4. Lưu model + CV metrics
  5. (Optional) SHAP report
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, List, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tqdm import tqdm

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments cho train.py."""
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình dự báo Revenue và Margin")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Đường dẫn tới file config.yaml",
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
# 3. Data Loading
# ---------------------------------------------------------------------------

def load_feature_table(processed_dir: Path) -> pd.DataFrame:
    """
    Load train_features.parquet từ thư mục processed.

    Parameters
    ----------
    processed_dir : Path
        Thư mục chứa train_features.parquet.

    Returns
    -------
    pd.DataFrame
        DataFrame đã sắp xếp theo cột date tăng dần.
    """
    path = processed_dir / "train_features.parquet"
    df = pd.read_parquet(path)
    df = df.sort_values("date").reset_index(drop=True)
    logger.info("Đã load train_features: %d dòng, %d cột", len(df), df.shape[1])
    return df


def get_feature_cols(
    df: pd.DataFrame,
    target_cols: List[str],
    date_col: str,
) -> List[str]:
    """
    Lấy danh sách cột feature (loại bỏ target và date).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame đầy đủ.
    target_cols : List[str]
        Danh sách cột target cần loại bỏ.
    date_col : str
        Tên cột date cần loại bỏ.

    Returns
    -------
    List[str]
        Danh sách tên cột dùng làm feature.
    """
    exclude = set(target_cols) | {date_col}
    feature_cols = [c for c in df.columns if c not in exclude]
    logger.info("Số feature columns: %d", len(feature_cols))
    return feature_cols


# ---------------------------------------------------------------------------
# 4. Model Registry
# ---------------------------------------------------------------------------

def get_model(model_name: str, model_cfg: dict, seed: int) -> Any:
    """
    Factory trả về model object từ tên và config.

    Parameters
    ----------
    model_name : str
        Tên model: lightgbm | xgboost | catboost | prophet.
    model_cfg : dict
        Hyperparameter dict từ config.yaml.
    seed : int
        Random seed để đảm bảo reproducibility.

    Returns
    -------
    Any
        Model object phù hợp với model_name.

    Raises
    ------
    ValueError
        Nếu model_name không được hỗ trợ.
    """
    if model_name == "lightgbm":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(**model_cfg, random_state=seed)

    elif model_name == "xgboost":
        from xgboost import XGBRegressor
        return XGBRegressor(**model_cfg, random_state=seed, tree_method="hist")

    elif model_name == "catboost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(**model_cfg, random_seed=seed)

    elif model_name == "prophet":
        from prophet import Prophet
        # Loại bỏ các key không thuộc Prophet constructor
        prophet_cfg = {k: v for k, v in model_cfg.items()}
        return Prophet(**prophet_cfg)

    else:
        raise ValueError(f"Model không hỗ trợ: {model_name}")


# ---------------------------------------------------------------------------
# 5. Cross-Validation — Expanding Window với Date Masking
# ---------------------------------------------------------------------------

def expanding_window_cv(
    df: pd.DataFrame,
    date_col: str,
    n_splits: int,
    min_train_days: int,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Tạo các fold Expanding Window CV bằng Date Masking.

    Lọc DataFrame bằng điều kiện trên cột date (không dùng integer index)
    để tránh lệch index khi tập dữ liệu có ngày bị thiếu.

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
    Tuple[pd.DataFrame, pd.DataFrame]
        Mỗi fold là cặp (df_train, df_val).
    """
    all_dates = df[date_col].sort_values().unique()
    total_days = len(all_dates)

    val_size = (total_days - min_train_days) // n_splits

    for fold in range(n_splits):
        train_end_idx = min_train_days + fold * val_size
        val_end_idx = train_end_idx + val_size

        train_cutoff = all_dates[train_end_idx - 1]
        val_start = all_dates[train_end_idx]
        val_end = all_dates[min(val_end_idx - 1, total_days - 1)]

        # DATE MASKING — lọc bằng điều kiện ngày, không dùng iloc
        df_train = df[df[date_col] <= train_cutoff].copy()
        df_val = df[
            (df[date_col] >= val_start) & (df[date_col] <= val_end)
        ].copy()

        logger.debug(
            "Fold %d: train đến %s | val %s – %s | train_size=%d val_size=%d",
            fold + 1,
            train_cutoff,
            val_start,
            val_end,
            len(df_train),
            len(df_val),
        )

        yield df_train, df_val


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Tính MAE, RMSE, R² giữa giá trị thực và dự báo.

    Parameters
    ----------
    y_true : np.ndarray
        Giá trị thực tế.
    y_pred : np.ndarray
        Giá trị dự báo.

    Returns
    -------
    Dict[str, float]
        Dictionary với keys: mae, rmse, r2.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    return {"mae": mae, "rmse": rmse, "r2": r2}


def _fit_predict_sklearn(
    model: Any,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    model_name: str,
    cfg: dict,
) -> np.ndarray:
    """
    Fit model sklearn-compatible và predict trên df_val.

    Parameters
    ----------
    model : Any
        Model object (LightGBM / XGBoost / CatBoost).
    df_train : pd.DataFrame
        Tập huấn luyện của fold.
    df_val : pd.DataFrame
        Tập validation của fold.
    feature_cols : List[str]
        Danh sách cột feature.
    target : str
        Tên cột target.
    model_name : str
        Tên model (để xử lý early stopping đúng cách).
    cfg : dict
        Config đầy đủ.

    Returns
    -------
    np.ndarray
        Mảng dự báo trên tập validation.
    """
    X_train = df_train[feature_cols]
    y_train = df_train[target]
    X_val = df_val[feature_cols]

    if model_name in ("lightgbm", "xgboost"):
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, df_val[target])],
            verbose=False
        )
    elif model_name == "catboost":
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, df_val[target]),
        )
    else:
        model.fit(X_train, y_train)

    return model.predict(X_val)


def _fit_predict_prophet(
    model: Any,
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    feature_cols: List[str],
    target: str,
) -> np.ndarray:
    """
    Fit Prophet và predict trên df_val.

    Rename cột thành 'ds'/'y', thêm external regressors.

    Parameters
    ----------
    model : Any
        Prophet model object.
    df_train : pd.DataFrame
        Tập huấn luyện với cột date và target.
    df_val : pd.DataFrame
        Tập validation.
    feature_cols : List[str]
        Danh sách cột regressor bổ sung.
    target : str
        Tên cột target.

    Returns
    -------
    np.ndarray
        Mảng dự báo trên tập validation.
    """
    # Chuẩn bị tập train cho Prophet
    train_prophet = df_train[["date", target] + feature_cols].rename(
        columns={"date": "ds", target: "y"}
    )
    for col in feature_cols:
        model.add_regressor(col)

    model.fit(train_prophet)

    # Chuẩn bị tập val
    val_prophet = df_val[["date"] + feature_cols].rename(columns={"date": "ds"})
    forecast = model.predict(val_prophet)
    return forecast["yhat"].values


def run_cv(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    cfg: dict,
) -> Dict[str, Any]:
    """
    Chạy Expanding Window CV, tính metrics cho từng fold và trung bình.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame với đầy đủ feature và target.
    feature_cols : List[str]
        Danh sách cột feature.
    target : str
        Tên cột target ('revenue' hoặc 'margin').
    cfg : dict
        Config đầy đủ từ config.yaml.

    Returns
    -------
    Dict[str, Any]
        Dictionary chứa metrics từng fold và mean/std.
    """
    cv_cfg = cfg["cv"]
    model_name: str = cfg["models"]["active"]
    seed: int = cfg["models"]["random_seed"]
    model_cfg: dict = cfg["models"][model_name]

    n_splits: int = cv_cfg["n_splits"]
    min_train_days: int = cv_cfg["min_train_days"]

    fold_metrics: List[Dict[str, float]] = []

    cv_gen = expanding_window_cv(df, "date", n_splits, min_train_days)

    for fold_idx, (df_train, df_val) in enumerate(
        tqdm(cv_gen, desc=f"CV [{target}]", total=n_splits)
    ):
        model = get_model(model_name, model_cfg, seed)

        if model_name == "prophet":
            y_pred = _fit_predict_prophet(model, df_train, df_val, feature_cols, target)
        else:
            y_pred = _fit_predict_sklearn(
                model, df_train, df_val, feature_cols, target, model_name, cfg
            )

        y_true = df_val[target].values
        metrics = compute_metrics(y_true, y_pred)
        fold_metrics.append(metrics)

        logger.info(
            "Fold %d [%s] — MAE=%.4f | RMSE=%.4f | R²=%.4f",
            fold_idx + 1,
            target,
            metrics["mae"],
            metrics["rmse"],
            metrics["r2"],
        )

    # Tính mean và std qua các fold
    keys = ["mae", "rmse", "r2"]
    mean_metrics = {k: float(np.mean([m[k] for m in fold_metrics])) for k in keys}
    std_metrics = {k: float(np.std([m[k] for m in fold_metrics])) for k in keys}

    logger.info(
        "CV [%s] MEAN — MAE=%.4f | RMSE=%.4f | R²=%.4f",
        target,
        mean_metrics["mae"],
        mean_metrics["rmse"],
        mean_metrics["r2"],
    )

    return {
        "target": target,
        "model": model_name,
        "fold_metrics": fold_metrics,
        "mean": mean_metrics,
        "std": std_metrics,
    }


# ---------------------------------------------------------------------------
# 6. Final Training
# ---------------------------------------------------------------------------

def train_final_model(
    df: pd.DataFrame,
    feature_cols: List[str],
    target: str,
    cfg: dict,
) -> Any:
    """
    Huấn luyện model cuối trên toàn bộ tập train.

    Parameters
    ----------
    df : pd.DataFrame
        Toàn bộ feature_table (chỉ phần train).
    feature_cols : List[str]
        Danh sách cột feature.
    target : str
        Tên cột target.
    cfg : dict
        Config đầy đủ.

    Returns
    -------
    Any
        Model đã fit.
    """
    model_name: str = cfg["models"]["active"]
    seed: int = cfg["models"]["random_seed"]
    model_cfg: dict = cfg["models"][model_name]

    model = get_model(model_name, model_cfg, seed)

    if model_name == "prophet":
        train_prophet = df[["date", target] + feature_cols].rename(
            columns={"date": "ds", target: "y"}
        )
        for col in feature_cols:
            model.add_regressor(col)
        model.fit(train_prophet)
    elif model_name in ("lightgbm", "xgboost", "catboost"):
        X = df[feature_cols]
        y = df[target]
        # Final train: không có eval_set — tắt early stopping bằng cách fit thẳng
        if model_name == "lightgbm":
            from lightgbm import LGBMRegressor
            # Override early_stopping_rounds để tránh yêu cầu eval_set
            final_cfg = {k: v for k, v in model_cfg.items() if k != "early_stopping_rounds"}
            model = LGBMRegressor(**final_cfg, random_state=seed)
        elif model_name == "xgboost":
            from xgboost import XGBRegressor
            final_cfg = {k: v for k, v in model_cfg.items() if k != "early_stopping_rounds"}
            model = XGBRegressor(**final_cfg, random_state=seed, tree_method="hist")
        elif model_name == "catboost":
            from catboost import CatBoostRegressor
            final_cfg = {k: v for k, v in model_cfg.items() if k != "early_stopping_rounds"}
            model = CatBoostRegressor(**final_cfg, random_seed=seed)
        model.fit(X, y)

    logger.info("Đã huấn luyện final model [%s] cho target '%s'", model_name, target)
    return model


# ---------------------------------------------------------------------------
# 7. Save / Load Utilities
# ---------------------------------------------------------------------------

def save_model(model: Any, path: Path) -> None:
    """
    Lưu model object vào file .pkl bằng pickle.

    Parameters
    ----------
    model : Any
        Model đã fit.
    path : Path
        Đường dẫn file đầu ra (.pkl).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Đã lưu model: %s", path)


def save_feature_cols(feature_cols: List[str], path: Path) -> None:
    """
    Lưu danh sách feature columns ra file JSON để inference dùng lại.

    Đảm bảo inference dùng đúng thứ tự và tập cột mà model đã train.

    Parameters
    ----------
    feature_cols : List[str]
        Danh sách tên cột feature theo thứ tự train.
    path : Path
        Đường dẫn file đầu ra (feature_cols.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(feature_cols, f, indent=2, ensure_ascii=False)
    logger.info("Đã lưu feature_cols (%d cột): %s", len(feature_cols), path)


def save_cv_metrics(metrics: Dict[str, Any], path: Path) -> None:
    """
    Lưu CV metrics ra file JSON.

    Parameters
    ----------
    metrics : Dict[str, Any]
        Dictionary metrics từ run_cv.
    path : Path
        Đường dẫn file đầu ra (.json).
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    logger.info("Đã lưu CV metrics: %s", path)


# ---------------------------------------------------------------------------
# 8. SHAP
# ---------------------------------------------------------------------------

def compute_shap(
    model: Any,
    X_sample: pd.DataFrame,
    cfg: dict,
    output_dir: Path,
    target: str,
) -> None:
    """
    Tính SHAP values và xuất report (html hoặc png).

    Chỉ hỗ trợ tree-based models (LightGBM, XGBoost, CatBoost).
    Prophet sẽ bị bỏ qua với cảnh báo.

    Parameters
    ----------
    model : Any
        Model đã fit.
    X_sample : pd.DataFrame
        Sample dữ liệu để tính SHAP (đã giới hạn số dòng).
    cfg : dict
        Config đầy đủ.
    output_dir : Path
        Thư mục lưu output.
    target : str
        Tên target (để đặt tên file).
    """
    import shap
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_name: str = cfg["models"]["active"]
    output_format: str = cfg["explainability"]["output_format"]

    if model_name == "prophet":
        logger.warning("SHAP không hỗ trợ Prophet — bỏ qua bước SHAP cho target '%s'", target)
        return

    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)

        # Top 10 feature theo mean |SHAP|
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top10_idx = np.argsort(mean_abs_shap)[::-1][:10]
        top10_features = [(X_sample.columns[i], mean_abs_shap[i]) for i in top10_idx]

        logger.info("Top 10 SHAP features [%s]:", target)
        for feat, val in top10_features:
            logger.info("  %-40s %.4f", feat, val)

        # Xuất plot
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)

        output_dir.mkdir(parents=True, exist_ok=True)

        if output_format == "html":
            # Lưu dưới dạng HTML qua matplotlib + base64
            import io
            import base64

            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode("utf-8")
            plt.close()

            html_path = output_dir / f"shap_report_{target}.html"
            html_content = f"""<!DOCTYPE html>
<html><head><title>SHAP Report — {target}</title></head>
<body>
<h2>SHAP Summary Plot — {target}</h2>
<img src="data:image/png;base64,{img_b64}" style="max-width:100%"/>
<h3>Top 10 Features (mean |SHAP|)</h3>
<ol>
{"".join(f"<li><b>{f}</b>: {v:.4f}</li>" for f, v in top10_features)}
</ol>
</body></html>"""
            html_path.write_text(html_content, encoding="utf-8")
            logger.info("Đã lưu SHAP report: %s", html_path)

        else:  # png
            png_path = output_dir / f"shap_report_{target}.png"
            plt.savefig(png_path, bbox_inches="tight", dpi=150)
            plt.close()
            logger.info("Đã lưu SHAP report: %s", png_path)

    except Exception as exc:
        logger.warning("Không thể tính SHAP cho target '%s': %s", target, exc)


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main() -> None:
    """
    Điểm vào chính của train.py.

    Luồng:
      1. Parse args, load config
      2. Load feature_table, lọc tập train
      3. CV cho revenue và margin
      4. Train final model cho cả hai target
      5. Save model + metrics
      6. (Optional) SHAP
    """
    args = parse_args()
    cfg = load_config(args.config)

    # Setup logging
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log"
    logging.basicConfig(
        level=getattr(logging, cfg["logging"]["level"]),
        format=cfg["logging"]["format"],
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    processed_dir = Path(cfg["paths"]["processed_dir"])
    model_dir = Path(cfg["paths"]["model_dir"])
    output_dir = Path(cfg["paths"]["output_dir"])

    target_revenue: str = cfg["data"]["target_revenue"]
    target_margin: str = cfg["data"]["target_margin"]
    target_cogs: str = cfg["data"]["target_cogs"]
    date_col: str = cfg["data"]["date_col"]

    TARGET_COLS = [target_revenue, target_margin, target_cogs]

    # Load và lọc tập train theo Date Masking
    df = load_feature_table(processed_dir)
    train_end = pd.Timestamp(cfg["data"]["train_end"])
    df_train_full = df[df["date"] <= train_end].copy()
    logger.info(
        "Tập train: %d dòng (%s → %s)",
        len(df_train_full),
        df_train_full["date"].min().date(),
        df_train_full["date"].max().date(),
    )

    feature_cols = get_feature_cols(df_train_full, TARGET_COLS, "date")

    model_name: str = cfg["models"]["active"]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = model_dir / f"{model_name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    all_cv_metrics: Dict[str, Any] = {}

    # ---- CV + Final training cho từng target ----
    for target in (target_revenue, target_margin):
        logger.info("=" * 60)
        logger.info("Bắt đầu CV cho target: %s", target)

        cv_result = run_cv(df_train_full, feature_cols, target, cfg)
        all_cv_metrics[target] = cv_result

        logger.info("Bắt đầu final training cho target: %s", target)
        final_model = train_final_model(df_train_full, feature_cols, target, cfg)

        model_filename = f"model_{target}.pkl"
        save_model(final_model, run_dir / model_filename)

        # SHAP
        if cfg["explainability"]["shap_enabled"] and model_name != "prophet":
            seed: int = cfg["models"]["random_seed"]
            shap_n: int = cfg["explainability"]["shap_sample_size"]
            X_sample = df_train_full[feature_cols].sample(
                n=min(shap_n, len(df_train_full)),
                random_state=seed,
            )
            compute_shap(final_model, X_sample, cfg, output_dir, target)

    # Lưu feature_cols để inference load lại đúng thứ tự
    save_feature_cols(feature_cols, run_dir / "feature_cols.json")

    # Lưu CV metrics
    save_cv_metrics(all_cv_metrics, run_dir / "cv_metrics.json")

    logger.info("Pipeline train hoàn tất. Model lưu tại: %s", run_dir)


if __name__ == "__main__":
    main()