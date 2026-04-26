"""
================================================================================
MODULE: TRAINING PIPELINE (HUẤN LUYỆN MÔ HÌNH CUỐI)
================================================================================

Mục đích:
    Nhận artifacts đã chuẩn bị từ validation.py, huấn luyện 2 model độc lập
    (Revenue và COGS), xuất model + feature importance phục vụ báo cáo datathon.

    Module này KHÔNG chạy lại CV, KHÔNG split data, KHÔNG tune hyperparams.
    Toàn bộ các bước đó đã hoàn tất ở validation.py và model_selection.py.
    train.py chỉ làm đúng 3 việc:
        Bước 1 — Load artifacts từ validation.py.
        Bước 2 — Train 2 model độc lập (Revenue + COGS).
        Bước 3 — Lưu model + feature_importance.

Lý do train riêng từng target:
    Revenue và COGS có đặc tính chuỗi thời gian khác nhau:
    - Revenue biến động mạnh theo mùa vụ, nhạy cảm với promo.
    - COGS ổn định hơn, gắn chặt với volume và cấu trúc chi phí.
    Optimal hyperparams của 2 target thường không trùng nhau.
    Train chung (MultiOutputRegressor) buộc dùng cùng params → suboptimal.

Lý do KHÔNG gọi run_validation() trong module này:
    Nếu muốn retrain với params mới, chỉ cần chạy lại train.py — không cần
    tốn thời gian chạy lại toàn bộ CV. Artifacts của validation.py
    (X_train, y_train, sample_weights) là stable trừ khi feature engineering
    thay đổi, nên tách biệt hoàn toàn 2 bước.

Cấu hình model:
    Sau khi chạy model_selection --tune → --apply, train.py tự đọc
    models/best_params.parquet. Không cần chỉnh chỗ nào khác.

Đầu ra — data/features/:
    model_Revenue.joblib        — Model đã fit cho target Revenue
    model_COGS.joblib           — Model đã fit cho target COGS
    feature_importance.parquet  — Tầm quan trọng feature (quan trọng cho báo cáo)
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.config import config

logger = config.get_logger(__name__)

# ── Constants (từ config) ──────────────────────────────────────────────────────
TARGETS      = config.TARGETS
RANDOM_STATE = config.RANDOM_STATE


# ==========================================
# CẤUHÌNH MODEL — CHỈNH SỬA TẠI ĐÂY SAU KHI ĐỌC comparison_summary.parquet
# ==========================================

def _build_chosen_models() -> dict:
    """
    Khởi tạo model cho từng target theo thứ tự ưu tiên:

        Ưu tiên 1 — models/best_params.parquet (tự động):
            Được ghi bởi model_selection.apply_best_params() sau khi Grid Search.
            Không cần chỉnh tay — chạy --tune → --apply → --train là xong.
            Persist cross-session: restart Python vẫn đọc được.

        Ưu tiên 2 — DEFAULT_PARAMS (hardcode fallback):
            Dùng khi chưa chạy Grid Search lần nào. Reasonable defaults dựa
            trên kinh nghiệm với time-series retail data.
            [Chỉnh tại đây nếu muốn override thủ công.]

    Returns:
        dict[str, model]: {'Revenue': model_obj, 'COGS': model_obj}

    Raises:
        ImportError: Nếu thư viện ML chưa được cài đặt.
    """
    import json as _json

    # ── DEFAULT PARAMS (Ưu tiên 3 — fallback) ────────────────────────────────
    # Chỉnh tại đây nếu muốn override thủ công, không cần Grid Search.
    DEFAULT_PARAMS = {
        'Revenue': {
            'model_class': 'LGBMRegressor',
            'params': {
                'n_estimators'     : 1000,
                'learning_rate'    : 0.03,
                'num_leaves'       : 127,
                'min_child_samples': 20,
                'subsample'        : 0.8,
                'colsample_bytree' : 0.8,
                'reg_alpha'        : 0.1,
                'reg_lambda'       : 0.1,
            }
        },
        'COGS': {
            'model_class': 'LGBMRegressor',
            'params': {
                # COGS ổn định hơn → shallower tree, ít regularization hơn
                'n_estimators'     : 800,
                'learning_rate'    : 0.03,
                'num_leaves'       : 63,
                'min_child_samples': 20,
                'subsample'        : 0.8,
                'colsample_bytree' : 0.8,
                'reg_alpha'        : 0.05,
                'reg_lambda'       : 0.05,
            }
        },
    }

    # ── Resolve params theo thứ tự ưu tiên ───────────────────────────────────
    # Ưu tiên 1: models/best_params.parquet (output của model_selection --apply)
    # Ưu tiên 2: DEFAULT_PARAMS hardcode (fallback khi chưa chạy grid search)
    best_params_path = config.MODELS / 'best_params.parquet'
    tuned = {}

    if best_params_path.exists():
        bp_df = pd.read_parquet(best_params_path)
        for _, row in bp_df.iterrows():
            tuned[row['target']] = {
                'model_class': row['model_class'],
                'params'     : _json.loads(row['params_json']),
            }
        logger.info(f"   Loaded tuned params từ {best_params_path}")
        logger.info(f"   Promoted at: {bp_df['promoted_at'].iloc[0]}")

    resolved = tuned if tuned else DEFAULT_PARAMS
    source   = f"best_params.parquet (Grid Search — promoted {bp_df['promoted_at'].iloc[0][:10]})" if tuned else "DEFAULT_PARAMS (fallback — chưa chạy grid search)"
    logger.info(f"   Params source: {source}")

    # ── Model registry ────────────────────────────────────────────────────────
    registry = {}
    try:
        from lightgbm import LGBMRegressor
        registry['LGBMRegressor'] = LGBMRegressor
    except ImportError:
        pass
    try:
        from xgboost import XGBRegressor
        registry['XGBRegressor'] = XGBRegressor
    except ImportError:
        pass
    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        registry['RandomForestRegressor']    = RandomForestRegressor
        registry['GradientBoostingRegressor']= GradientBoostingRegressor
    except ImportError:
        pass

    if not registry:
        raise ImportError(
            "Không có thư viện ML nào. Cài: pip install lightgbm xgboost scikit-learn"
        )

    # ── Khởi tạo model cho từng target ───────────────────────────────────────
    models = {}
    for target in TARGETS:
        spec         = resolved.get(target, DEFAULT_PARAMS.get(target, {}))
        class_name   = spec.get('model_class', 'LGBMRegressor')
        params       = spec.get('params', {})

        if class_name not in registry:
            raise ImportError(
                f"Model class '{class_name}' cho target [{target}] chưa được cài đặt."
            )

        model_cls = registry[class_name]
        try:
            model = model_cls(
                **params,
                random_state = RANDOM_STATE,
                n_jobs       = -1,
                verbosity    = -1,
            )
        except TypeError:
            # Fallback: model không nhận verbosity hoặc n_jobs
            try:
                model = model_cls(**params, random_state=RANDOM_STATE, n_jobs=-1)
            except TypeError:
                model = model_cls(**params, random_state=RANDOM_STATE)

        models[target] = model
        logger.info(f"   [{target}] {class_name} | params={params}")

    return models


# ==========================================
# BƯỚC 1: LOAD ARTIFACTS
# ==========================================

def _load_training_artifacts() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Đọc X_train, y_train, sample_weights từ validation artifacts.

    Không rebuild, không retransform — đọc thẳng parquet đã sẵn sàng.
    Nếu artifacts chưa tồn tại, raise lỗi rõ ràng với hướng dẫn khắc phục.

    Returns:
        X_train (pd.DataFrame)   : Feature matrix đã impute + scale.
        y_train (pd.DataFrame)   : Target vector (Date, Revenue, COGS).
        sample_weights (np.ndarray): Sample weight từng ngày train.

    Raises:
        FileNotFoundError: Nếu một trong các artifact chưa tồn tại.
    """
    required = {
        'X_train'       : config.FEATURES / 'X_train.parquet',
        'y_train'       : config.FEATURES / 'y_train.parquet',
        'sample_weights': config.FEATURES / 'sample_weights.parquet',
    }

    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Chưa tìm thấy artifacts: {missing}.\n"
            f"Hãy chạy validation pipeline trước: python main.py --feat (rồi validation)."
        )

    X_train        = pd.read_parquet(required['X_train'])
    y_train        = pd.read_parquet(required['y_train'])
    sw_df          = pd.read_parquet(required['sample_weights'])
    sample_weights = sw_df['sample_weight'].values

    logger.info(f"   X_train        : {X_train.shape}")
    logger.info(f"   y_train        : {y_train.shape}")
    logger.info(f"   sample_weights : shape={sample_weights.shape} | "
                f"min={sample_weights.min():.3f} | mean={sample_weights.mean():.3f}")

    return X_train, y_train, sample_weights


# ==========================================
# BƯỚC 2: TRAIN MỖI TARGET ĐỘC LẬP
# ==========================================

def _train_single_target(
    model,
    X_train        : pd.DataFrame,
    y_train        : pd.Series,
    sample_weights : np.ndarray,
    target_name    : str,
) -> object:
    """
    Fit một model cho một target cụ thể với sample weights.

    Truyền sample_weight qua fit() thay vì constructor để tương thích với
    cả LightGBM, XGBoost và sklearn API. Với sklearn models không hỗ trợ
    sample_weight, sẽ fallback về fit không weight và log cảnh báo.

    Args:
        model          : Model instance chưa fit (từ _build_chosen_models()).
        X_train        : Feature matrix.
        y_train        : Target series (1 cột).
        sample_weights : Weight từng mẫu.
        target_name    : Tên target — dùng cho log và tên file.

    Returns:
        Model đã fit.
    """
    logger.info(f"   Fit model cho target: [{target_name}]")
    logger.info(f"   Samples: {len(X_train)} | Features: {X_train.shape[1]}")

    try:
        model.fit(X_train, y_train, sample_weight=sample_weights)
        logger.info(f"   [{target_name}] Fit thành công ✅ (có sample_weight)")
    except TypeError:
        # Fallback: một số sklearn models không nhận sample_weight trong fit()
        logger.warning(
            f"   [{target_name}] Model không hỗ trợ sample_weight → fit không weight."
        )
        model.fit(X_train, y_train)
        logger.info(f"   [{target_name}] Fit thành công ✅ (không sample_weight)")

    return model


# ==========================================
# BƯỚC 3: XUẤT FEATURE IMPORTANCE
# ==========================================

def _extract_feature_importance(
    models       : dict,
    feature_cols : list[str],
) -> pd.DataFrame:
    """
    Tổng hợp feature importance từ tất cả model đã train.

    Hỗ trợ 2 loại importance API:
        - feature_importances_ (LightGBM, XGBoost, RandomForest, GBM): gain-based.
        - coef_ (LinearRegression, Ridge, Lasso): hệ số hồi quy (abs value).

    Nếu model không có cả hai, cột importance sẽ là NaN — không crash.

    Cột đầu ra:
        feature_name  : Tên feature.
        importance_{target} : Score importance cho từng target.
        importance_mean     : Trung bình giữa các target (dùng cho báo cáo tổng).
        rank_mean           : Rank theo importance_mean (1 = quan trọng nhất).
        group               : Nhóm feature (calendar / yoy / seasonal / promo),
                              join từ feature_metadata.parquet nếu có.

    Args:
        models       (dict[str, model]): Model đã fit theo target.
        feature_cols (list[str])       : Tên feature theo đúng thứ tự trong X_train.

    Returns:
        pd.DataFrame: Feature importance đã rank, sẵn sàng cho báo cáo.
    """
    importance_df = pd.DataFrame({'feature_name': feature_cols})

    for target, model in models.items():
        col_name = f'importance_{target}'
        if hasattr(model, 'feature_importances_'):
            importance_df[col_name] = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importance_df[col_name] = np.abs(model.coef_)
        else:
            logger.warning(f"   [{target}] Model không có feature_importances_ hoặc coef_.")
            importance_df[col_name] = np.nan

    # Tính trung bình và rank
    imp_cols = [f'importance_{t}' for t in models.keys() if f'importance_{t}' in importance_df.columns]
    importance_df['importance_mean'] = importance_df[imp_cols].mean(axis=1)
    importance_df = importance_df.sort_values('importance_mean', ascending=False).reset_index(drop=True)
    importance_df.insert(1, 'rank_mean', range(1, len(importance_df) + 1))

    # Gắn thêm nhóm feature nếu có metadata
    meta_path = config.FEATURES / 'feature_metadata.parquet'
    if meta_path.exists():
        meta = pd.read_parquet(meta_path)[['feature_name', 'group']]
        importance_df = importance_df.merge(meta, on='feature_name', how='left')
        importance_df['group'] = importance_df['group'].fillna('unknown')
    else:
        logger.warning("   feature_metadata.parquet chưa có — cột 'group' sẽ bị bỏ qua.")

    logger.info(f"\n  Top 10 features theo importance_mean:")
    for _, row in importance_df.head(10).iterrows():
        imp_vals = " | ".join(
            f"{t}={row[f'importance_{t}']:>8.1f}"
            for t in models.keys()
            if f'importance_{t}' in row.index
        )
        group = row.get('group', '?')
        logger.info(f"    #{int(row['rank_mean']):>3} [{group:<8}] {row['feature_name']:<40} {imp_vals}")

    return importance_df


# ==========================================
# ORCHESTRATOR
# ==========================================

def run_train(force_retrain: bool = False) -> dict:
    """
    Entry point chính — thực thi toàn bộ Training Pipeline.

    - force_retrain=False (default): Bỏ qua nếu model_Revenue.joblib đã tồn tại.
    - force_retrain=True           : Train lại từ đầu dù artifact đã có.

    Thứ tự:
        1. Kiểm tra artifact đã tồn tại chưa.
        2. Load X_train, y_train, sample_weights.
        3. Khởi tạo model theo CHOSEN_MODELS.
        4. Train riêng từng target.
        5. Xuất model.joblib + feature_importance.parquet.

    Args:
        force_retrain (bool): Mặc định False để tránh train lại vô ý.

    Returns:
        dict[str, model]: Model đã fit theo từng target.

    Raises:
        FileNotFoundError: Nếu artifacts chưa tồn tại.
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU TRAINING PIPELINE")
    logger.info("=" * 60)

    rev_model_path = config.FEATURES / 'model_Revenue.joblib'

    if not force_retrain and rev_model_path.exists():
        logger.info(
            "Model đã tồn tại. Bỏ qua để tiết kiệm thời gian "
            "(đặt force_retrain=True để train lại)."
        )
        # Load và trả về model đã có để inference có thể dùng
        fitted_models = {}
        for target in TARGETS:
            path = config.FEATURES / f'model_{target}.joblib'
            if path.exists():
                fitted_models[target] = joblib.load(path)
                logger.info(f"   Load model [{target}] từ cache ✅")
        return fitted_models

    # ── Bước 1: Load artifacts ────────────────────────────────────────────────
    logger.info("\nBước 1 — Load training artifacts...")
    X_train, y_train, sample_weights = _load_training_artifacts()

    feature_cols = list(X_train.columns)

    # ── Bước 2: Khởi tạo model đã chọn ───────────────────────────────────────
    logger.info("\nBước 2 — Khởi tạo model (từ best_params.parquet hoặc DEFAULT_PARAMS)...")
    chosen_models = _build_chosen_models()
    logger.info(f"   Targets sẽ train: {list(chosen_models.keys())}")

    # ── Bước 3: Train từng target ─────────────────────────────────────────────
    logger.info("\nBước 3 — Train model cho từng target...")
    fitted_models = {}

    for target in TARGETS:
        if target not in chosen_models:
            logger.warning(f"   [{target}] Không có trong resolved params — bỏ qua.")
            continue
        if target not in y_train.columns:
            logger.error(f"   [{target}] Không có trong y_train — bỏ qua.")
            continue

        y_target = y_train[target].values
        model    = chosen_models[target]

        fitted_model = _train_single_target(
            model          = model,
            X_train        = X_train,
            y_train        = y_target,
            sample_weights = sample_weights,
            target_name    = target,
        )
        fitted_models[target] = fitted_model

    # ── Bước 4: Xuất model.joblib + feature_importance ────────────────────────
    logger.info("\nBước 4 — Lưu model và feature importance...")
    config.FEATURES.mkdir(parents=True, exist_ok=True)

    for target, model in fitted_models.items():
        out_path = config.FEATURES / f'model_{target}.joblib'
        joblib.dump(model, out_path)
        logger.info(f"   Đã lưu: {out_path}")

    importance_df = _extract_feature_importance(fitted_models, feature_cols)
    importance_path = config.FEATURES / 'feature_importance.parquet'
    importance_df.to_parquet(importance_path, index=False)
    logger.info(f"   Đã lưu: {importance_path}")

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING PIPELINE HOÀN TẤT ✅")
    logger.info(f"  Models    : {[f'model_{t}.joblib' for t in fitted_models]}")
    logger.info(f"  Importance: feature_importance.parquet ({len(importance_df)} features)")
    logger.info(f"  Output    : {config.FEATURES}")
    logger.info("=" * 60)

    return fitted_models