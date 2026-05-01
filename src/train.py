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
import xgboost as xgb
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
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Chạy nhanh để test lỗi runtime",
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

    # FIX-2: fix val_size to match real forecast horizon (~1.5 years)
    VAL_SIZE = 90
    MIN_TRAIN_SIZE = 365

    for fold in range(n_splits):
        val_end_idx = total_days - fold * VAL_SIZE
        val_start_idx = val_end_idx - VAL_SIZE
        train_end_idx = val_start_idx
        
        if train_end_idx < MIN_TRAIN_SIZE:
            continue
            
        train_cutoff = all_dates[train_end_idx - 1]
        val_start = all_dates[val_start_idx]
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
        logger.info(
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
    X_train = df_train[feature_cols].copy()
    y_train = df_train[target]
    X_val = df_val[feature_cols].copy()

    sw_train = X_train["sample_weight"].values if "sample_weight" in X_train.columns else None
    sw_val = X_val["sample_weight"].values if "sample_weight" in X_val.columns else None

    drop_cols = ["sample_weight", "is_covid_period"]
    X_train = X_train.drop(columns=[c for c in drop_cols if c in X_train.columns])
    X_val = X_val.drop(columns=[c for c in drop_cols if c in X_val.columns])

    # FIX-3a: log-transform target to stabilise variance
    _use_log = target in ("revenue", "cogs")
    y_train   = np.log1p(y_train)          if _use_log else y_train
    y_val_log = np.log1p(df_val[target])   if _use_log else df_val[target]

    if model_name in ("lightgbm", "xgboost"):
        fit_kwargs = {}
        if sw_train is not None:
            fit_kwargs["sample_weight"] = sw_train
            fit_kwargs["sample_weight_eval_set"] = [sw_val]
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val_log)],
            verbose=False,
            **fit_kwargs
        )
    elif model_name == "catboost":
        fit_kwargs = {}
        if sw_train is not None:
            fit_kwargs["sample_weight"] = sw_train
        model.fit(
            X_train,
            y_train,
            eval_set=(X_val, y_val_log),
            **fit_kwargs
        )
    else:
        fit_kwargs = {}
        if sw_train is not None:
            fit_kwargs["sample_weight"] = sw_train
        model.fit(X_train, y_train, **fit_kwargs)

    preds = model.predict(X_val)
    # FIX-3b: inverse log-transform predictions
    preds = np.expm1(preds) if _use_log else preds
    
    if target == "margin":
        preds = np.clip(preds, 0.0, 1.0)
        
    return preds


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
        if model_name == "prophet":
            model = get_model(model_name, model_cfg, seed)
            y_pred = _fit_predict_prophet(model, df_train, df_val, feature_cols, target)
        else:
            model = get_model(model_name, model_cfg, seed)
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


class HybridRegressor:
    """
    Sklearn-compatible Hybrid: TrendModel (Ridge/Prophet) +
    XGBoost Residual Learner.

    fit() nhận full train DataFrame, tự tách trend và residual.
    predict() trả về trend_pred + residual_pred dưới dạng np.ndarray.
    Toàn bộ caller (CV loop, inference) chỉ thấy .fit() và .predict().
    """

    def __init__(self, cfg: dict) -> None:
        """
        Khởi tạo HybridRegressor.

        Parameters
        ----------
        cfg : dict
            Full config dict đọc từ config.yaml.
        """
        self.cfg = cfg
        self.trend_model_ = None
        self.residual_model_ = None
        self.residual_cols_: List[str] = []
        self.trend_model_name = cfg["hybrid"]["trend_model"]
        self.train_date_min_: pd.Timestamp | None = None
        self.target_: str | None = None

    def _build_trend_model(self) -> Any:
        """
        Khởi tạo model cho phần trend.

        Returns
        -------
        Any
            Model object (Ridge hoặc Prophet).
        """
        name = self.trend_model_name
        if name == "ridge":
            from sklearn.linear_model import Ridge
            alpha = self.cfg["hybrid"]["trend"]["ridge"]["alpha"]
            return Ridge(alpha=alpha)
        elif name == "prophet":
            from prophet import Prophet
            params = self.cfg["hybrid"]["trend"]["prophet"]
            return Prophet(**params)
        else:
            raise ValueError(f"Trend model không hợp lệ: {name}")

    def _fit_trend(self, df: pd.DataFrame, target: str) -> None:
        """
        Huấn luyện model phần trend.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame huấn luyện.
        target : str
            Tên cột mục tiêu.
        """
        self.target_ = target
        self.train_date_min_ = df["date"].min()

        y = df[target].values

        if self.trend_model_name == "ridge":
            trend_index = (df["date"] - self.train_date_min_).dt.days.values
            X_trend = pd.DataFrame({"trend_index": trend_index})
            X_trend["trend_index_sq"] = X_trend["trend_index"] ** 2
            
            X_trend["sin_365_1"] = np.sin(2 * np.pi * trend_index / 365)
            X_trend["cos_365_1"] = np.cos(2 * np.pi * trend_index / 365)
            X_trend["sin_365_2"] = np.sin(4 * np.pi * trend_index / 365)
            X_trend["cos_365_2"] = np.cos(4 * np.pi * trend_index / 365)

            self.trend_model_.fit(X_trend, y)
        elif self.trend_model_name == "prophet":
            df_prophet = df[["date", target]].copy()
            df_prophet = df_prophet.rename(columns={"date": "ds"})
            df_prophet["y"] = y
            self.trend_model_.fit(df_prophet)

    def _predict_trend(self, df: pd.DataFrame) -> np.ndarray:
        """
        Dự báo bằng model phần trend.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame cần dự báo.

        Returns
        -------
        np.ndarray
            Mảng chứa giá trị dự báo từ trend.
        """
        if self.trend_model_name == "ridge":
            trend_index = (df["date"] - self.train_date_min_).dt.days.values
            X_trend = pd.DataFrame({"trend_index": trend_index})
            X_trend["trend_index_sq"] = X_trend["trend_index"] ** 2
            
            X_trend["sin_365_1"] = np.sin(2 * np.pi * trend_index / 365)
            X_trend["cos_365_1"] = np.cos(2 * np.pi * trend_index / 365)
            X_trend["sin_365_2"] = np.sin(4 * np.pi * trend_index / 365)
            X_trend["cos_365_2"] = np.cos(4 * np.pi * trend_index / 365)

            raw_preds = self.trend_model_.predict(X_trend)
        elif self.trend_model_name == "prophet":
            future = df[["date"]].copy()
            future = future.rename(columns={"date": "ds"})
            forecast = self.trend_model_.predict(future)
            raw_preds = forecast["yhat"].values

        return raw_preds

    def _build_residual_model(self) -> Any:
        """
        Khởi tạo model cho phần residual.

        Returns
        -------
        Any
            Model object XGBoost.
        """
        from xgboost import XGBRegressor
        params = self.cfg["models"]["xgboost"]
        final_params = {k: v for k, v in params.items() if k != "early_stopping_rounds"}
        final_params["early_stopping_rounds"] = params.get("early_stopping_rounds", 50)
        return XGBRegressor(**final_params, random_state=self.cfg["models"]["random_seed"], tree_method="hist")

    def _get_residual_feature_cols(self, all_feature_cols: List[str]) -> List[str]:
        """
        Lọc các cột feature dành cho phần residual.

        Parameters
        ----------
        all_feature_cols : List[str]
            Danh sách tất cả các cột feature.

        Returns
        -------
        List[str]
            Danh sách các cột feature được giữ lại cho residual.
        """
        allowed_groups = self.cfg["hybrid"]["residual"]["feature_groups"]
        
        prefix_map = self.cfg["hybrid"]["residual"].get("feature_prefixes", {})
        
        allowed_prefixes = []
        for group in allowed_groups:
            if group in prefix_map:
                allowed_prefixes.extend(prefix_map[group])
            else:
                logger.warning("Feature group '%s' not found in feature_prefixes config.", group)

        filtered_cols = [
            col for col in all_feature_cols 
            if any(col.startswith(prefix) for prefix in allowed_prefixes)
        ]
        
        logger.info("Số cột residual được giữ lại: %d", len(filtered_cols))
        return filtered_cols

    def fit(self, df: pd.DataFrame, target: str, feature_cols: List[str], eval_df: pd.DataFrame | None = None) -> "HybridRegressor":
        """
        Huấn luyện mô hình Hybrid.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame huấn luyện.
        target : str
            Tên cột mục tiêu.
        feature_cols : List[str]
            Danh sách tất cả feature columns.
        eval_df : pd.DataFrame | None, optional
            DataFrame dùng cho validation, by default None.

        Returns
        -------
        HybridRegressor
            Mô hình đã được huấn luyện.
        """
        self.trend_model_ = self._build_trend_model()
        self._fit_trend(df, target)
        trend_preds = self._predict_trend(df)
        
        residuals = df[target].values - trend_preds
        
        res_cfg = self.cfg["hybrid"]["residual_clip"]
        if res_cfg["enabled"]:
            lo = np.quantile(residuals, res_cfg["lower_quantile"])
            hi = np.quantile(residuals, res_cfg["upper_quantile"])
            residuals = np.clip(residuals, lo, hi)
            logger.info("Residual clip: [%.2f, %.2f]", lo, hi)
            
        self.residual_cols_ = self._get_residual_feature_cols(feature_cols)
        X_resid = df[self.residual_cols_]
        
        if eval_df is not None:
            eval_trend = self._predict_trend(eval_df)
            eval_resid = eval_df[target].values - eval_trend
            eval_set = [(eval_df[self.residual_cols_], eval_resid)]
        else:
            eval_set = None
            
        self.residual_model_ = self._build_residual_model()
        
        if eval_set is not None:
            self.residual_model_.fit(X_resid, residuals, eval_set=eval_set, verbose=False)
        else:
            self.residual_model_.fit(X_resid, residuals, verbose=False)
            
        logger.info("Fit summary: trend_model=%s, residual_features=%d", self.trend_model_name, len(self.residual_cols_))
        return self

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """
        Dự báo.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame cần dự báo.

        Returns
        -------
        np.ndarray
            Mảng chứa giá trị dự báo.
        """
        trend_preds = self._predict_trend(df)
        resid_preds = self.residual_model_.predict(df[self.residual_cols_])
        
        final = trend_preds + resid_preds
        
        logger.info(
            "Predict mean - trend: %.2f | residual: %.2f | final: %.2f",
            float(np.mean(trend_preds)), float(np.mean(resid_preds)), float(np.mean(final))
        )
        return final


# ---------------------------------------------------------------------------
# 6. Final Training
# ---------------------------------------------------------------------------

def train_final_model(df: pd.DataFrame, feature_cols: list, target: str, cfg: dict) -> tuple:
    df = df.copy()

    hybrid_enabled: bool = cfg.get("hybrid", {}).get("enabled", False)

    if hybrid_enabled:
        # 30-day micro-holdout for XGBoost residual early stopping
        holdout_cutoff = df["date"].max() - pd.Timedelta(days=30)
        train_df = df[df["date"] <= holdout_cutoff].copy()
        eval_df  = df[df["date"] >  holdout_cutoff].copy()

        model = HybridRegressor(cfg)
        model.fit(train_df, target, feature_cols, eval_df=eval_df)

    else:
        # Pure XGBoost path (keep for hybrid.enabled: false fallback)
        if target in ("revenue", "cogs"):
            df[target] = np.log1p(df[target])

        holdout_cutoff = df["date"].max() - pd.Timedelta(days=30)
        train_df = df[df["date"] <= holdout_cutoff]
        val_df   = df[df["date"] >  holdout_cutoff]

        X_train, y_train = train_df[feature_cols], train_df[target]
        X_val,   y_val   = val_df[feature_cols],   val_df[target]

        xgb_params = cfg["models"]["xgboost"].copy()
        early_stopping_rounds = xgb_params.pop("early_stopping_rounds", 50)
        verbosity = xgb_params.pop("verbosity", 0)

        model = xgb.XGBRegressor(
            **xgb_params,
            early_stopping_rounds=early_stopping_rounds,
            verbosity=verbosity,
            random_state=cfg["models"]["random_seed"],
        )
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

    return model, feature_cols


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

    if getattr(args, "smoke_test", False):
        logger.info("SMOKE TEST MODE: Giảm dữ liệu và epoch")
        df_train_full = df_train_full.tail(1000).copy()
        cfg["cv"]["n_splits"] = 1
        if "xgboost" in cfg["models"]:
            cfg["models"]["xgboost"]["n_estimators"] = 10
        if "catboost" in cfg["models"]:
            cfg["models"]["catboost"]["iterations"] = 10
        if "lightgbm" in cfg["models"]:
            cfg["models"]["lightgbm"]["n_estimators"] = 10

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
        final_model, feature_cols = train_final_model(df_train_full, feature_cols, target, cfg)

        model_filename = f"model_{target}.pkl"
        save_model(final_model, run_dir / model_filename)

        # SHAP
        if cfg["explainability"]["shap_enabled"]:
            if model_name != "prophet":
                shap_model = final_model
                shap_cols = feature_cols
                valid_model = True
            else:
                valid_model = False

            if valid_model:
                seed: int = cfg["models"]["random_seed"]
                shap_n: int = cfg["explainability"]["shap_sample_size"]
                X_sample = df_train_full[shap_cols].sample(
                    n=min(shap_n, len(df_train_full)),
                    random_state=seed,
                )
                compute_shap(shap_model, X_sample, cfg, output_dir, target)

    # Lưu feature_cols để inference load lại đúng thứ tự
    save_feature_cols(feature_cols, run_dir / "feature_cols.json")

    # Lưu CV metrics
    save_cv_metrics(all_cv_metrics, run_dir / "cv_metrics.json")

    # Lưu train_stats.json cho post-processing
    train_stats = {
        "revenue_p99": float(df_train_full[target_revenue].quantile(0.99)),
        "revenue_p05": float(df_train_full[target_revenue].quantile(0.05)),
        "margin_mean": float(df_train_full[target_margin].mean())
    }
    with open(processed_dir / "train_stats.json", "w", encoding="utf-8") as f:
        import json
        json.dump(train_stats, f, indent=2)
    logger.info("Đã lưu train_stats.json")

    logger.info("Pipeline train hoàn tất. Model lưu tại: %s", run_dir)


if __name__ == "__main__":
    main()