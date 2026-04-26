"""
================================================================================
MODULE: VALIDATION PIPELINE
================================================================================

Mục đích:
    Nhận X_full + y_full từ feature_eng.py, thực hiện toàn bộ các bước cần
    "biết về split" trước khi data đến tay train.py.

    Trách nhiệm theo thứ tự:
        1. Leakage Check    : Hard stop nếu raw target lọt vào feature matrix.
        2. Sample Weights   : Exponential time decay + COVID period penalty.
        3. Walk-Forward CV  : 4 fold expanding window — đánh giá model stability.
        4. Final Split      : Impute + Scale + Export artifacts cho train.py.

Thiết kế Walk-Forward CV (Expanding Window):
    Fold sau luôn có nhiều train data hơn fold trước.
    Gap 6 tháng giữa train_end và val_start để simulate điều kiện test thực
    (lag_728 là safe lag nhỏ nhất — fold không được "dễ" hơn test).

    Sơ đồ (val_months=6, gap_months=6, n_folds=4):
        Fold 1: Train [TRAIN_START → 2019-06] | Val [2020-01 → 2020-06]
        Fold 2: Train [TRAIN_START → 2020-06] | Val [2021-01 → 2021-06]
        Fold 3: Train [TRAIN_START → 2021-06] | Val [2022-01 → 2022-06]
        Fold 4: Train [TRAIN_START → 2022-06] | Val [2023-01 → 2023-06]
                                                 ↑ gần test thực nhất

    rebuild_seasonal_per_fold=True (default):
        Nhóm 3 seasonal stats được rebuild với train_mask của từng fold.
        Tránh fold đầu được "tặng thêm" seasonal info từ các năm chưa thuộc train.

Sample Weights:
    weight = 2^(−days_from_train_end / halflife) × covid_multiplier
    COVID penalty: nhân thêm covid_penalty (0.4) cho 2019-01 → 2021-12.
    Lý do không bỏ hẳn data COVID: vẫn chứa seasonal pattern tháng/quý.
    Normalize về [min_weight, 1.0] để tránh numerical instability.

Đầu ra — data/features/:
    X_train.parquet        — Feature matrix train (đã impute + scale)
    X_test.parquet         — Feature matrix test  (đã impute + scale, có cột 'Date')
    y_train.parquet        — Target train
    sample_weights.parquet — Weight từng ngày train (Date, sample_weight)
    cv_results.parquet     — Metrics từng fold × target
    scaler.joblib          — StandardScaler fit trên full train
    train_medians.joblib   — Median fit trên full train (dùng cho inference)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.config import config
from src import feature_eng

logger = config.get_logger(__name__)

# ── Constants (từ config) ──────────────────────────────────────────────────────
TRAIN_START          = config.TRAIN_START
TRAIN_END            = config.TRAIN_END
TEST_START           = config.TEST_START
TEST_END             = config.TEST_END
TARGETS              = config.TARGETS
SAFE_LAG_DAYS        = config.SAFE_LAG_DAYS
WEIGHT_HALFLIFE_DAYS = config.WEIGHT_HALFLIFE_DAYS
WEIGHT_COVID_PENALTY = config.WEIGHT_COVID_PENALTY
WEIGHT_MIN           = config.WEIGHT_MIN
COVID_START          = config.COVID_START
COVID_END            = config.COVID_END

# Safe suffixes cho leakage check — phải khớp với tất cả feature names trong feature_eng
_SAFE_SUFFIXES = (
    [f'lag{l}' for l in SAFE_LAG_DAYS]
    + ['yoy_growth', 'roll', 'trend_slope', 'vol', 'cv', 'signal_quality',
       'seasonal_idx', 'by_month', 'by_quarter', 'by_dow', 'by_week']
)


# ==========================================
# 1. LEAKAGE CHECK
# ==========================================

def check_leakage(X: pd.DataFrame) -> None:
    """
    Hard stop nếu raw target (t=0) lọt vào feature matrix.

    Logic: cột nào chứa tên target (Revenue/COGS, case-insensitive) nhưng
    KHÔNG chứa safe suffix nào → nghi ngờ leakage → raise ValueError.

    Safe suffixes = tất cả các dạng feature hợp lệ từ feature_eng:
        lag364, lag728, yoy_growth, roll*, trend_slope,
        vol*, cv*, signal_quality*, seasonal_idx, by_month*, ...

    Args:
        X (pd.DataFrame): Feature matrix (có thể có cột 'Date').

    Raises:
        ValueError: Nếu phát hiện bất kỳ cột nào nghi ngờ leakage.
    """
    leakage_cols = [
        c for c in X.columns
        if c != 'Date'
        and any(t.lower() in c.lower() for t in TARGETS)
        and not any(s in c.lower() for s in _SAFE_SUFFIXES)
    ]
    if leakage_cols:
        raise ValueError(
            f"\n{'=' * 60}\n"
            f"[LEAKAGE DETECTED] Cột expose raw target:\n{leakage_cols}\n"
            f"Chỉ cho phép lag >= {min(SAFE_LAG_DAYS)} ngày.\n"
            f"{'=' * 60}"
        )
    logger.info("   Leakage check passed ✅")


# ==========================================
# 2. SAMPLE WEIGHTS
# ==========================================

def compute_sample_weights(dates: pd.Series) -> np.ndarray:
    """
    Tính sample weight cho từng ngày train.

    Công thức:
        weight_time    = 2^(−days_from_train_end / HALFLIFE)
        covid_mult     = WEIGHT_COVID_PENALTY nếu trong COVID period, else 1.0
        raw_weight     = weight_time × covid_mult
        final_weight   = normalize về [WEIGHT_MIN, 1.0]

    Ví dụ với halflife=365, covid_penalty=0.4:
        2022-12-31 : weight_time=1.00, mult=1.0 → raw=1.00 → final≈1.00
        2021-06-15 : weight_time=0.71, mult=0.4 → raw=0.28 → final≈0.32
        2019-01-01 : weight_time=0.50, mult=0.4 → raw=0.20 → final≈0.24
        2016-01-01 : weight_time=0.25, mult=1.0 → raw=0.25 → final≈0.29
        2012-07-04 : weight_time=0.08, mult=1.0 → raw=0.08 → final≈0.05 (min)

    Args:
        dates (pd.Series): Ngày của từng mẫu train, sau khi đã filter cold start.

    Returns:
        np.ndarray: shape (n_samples,), giá trị trong [WEIGHT_MIN, 1.0].
    """
    dates       = pd.to_datetime(dates).reset_index(drop=True)
    train_end   = pd.Timestamp(TRAIN_END)
    covid_start = pd.Timestamp(COVID_START)
    covid_end   = pd.Timestamp(COVID_END)

    days_from_end = (train_end - dates).dt.days.clip(lower=0)
    weight_time   = np.power(2.0, -(days_from_end / WEIGHT_HALFLIFE_DAYS))

    is_covid   = ((dates >= covid_start) & (dates <= covid_end)).astype(float)
    covid_mult = 1.0 - (1.0 - WEIGHT_COVID_PENALTY) * is_covid

    raw = weight_time * covid_mult

    w_min, w_max = raw.min(), raw.max()
    if w_max > w_min:
        weights = WEIGHT_MIN + (1.0 - WEIGHT_MIN) * (raw - w_min) / (w_max - w_min)
    else:
        weights = np.ones(len(raw))

    logger.info(
        f"   Sample weights: min={weights.min():.3f} | "
        f"mean={weights.mean():.3f} | max={weights.max():.3f}"
    )
    covid_mask = is_covid.astype(bool)
    if covid_mask.any():
        logger.info(
            f"   COVID weight avg: {weights[covid_mask].mean():.3f} | "
            f"Non-COVID avg: {weights[~covid_mask].mean():.3f}"
        )
    return weights.values


# ==========================================
# 3. IMPUTE + SCALE (fit/transform tách biệt)
# ==========================================

def _fit_imputer(X_train: pd.DataFrame) -> pd.Series:
    """Fit median trên train. Không bao giờ nhìn vào val/test."""
    return X_train.median()


def _apply_imputer(X: pd.DataFrame, medians: pd.Series) -> pd.DataFrame:
    """fillna bằng train medians đã fit."""
    return X.fillna(medians.to_dict()).fillna(0)


def _fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """Fit StandardScaler trên train đã impute."""
    return StandardScaler().fit(X_train)


def _apply_scaler(X: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """Transform bằng scaler đã fit."""
    return pd.DataFrame(scaler.transform(X), columns=X.columns, index=X.index)


# ==========================================
# 4. WALK-FORWARD CROSS VALIDATION
# ==========================================

def _build_cv_folds(
    dates      : pd.Series,
    n_folds    : int = 4,
    val_months : int = 6,
    gap_months : int = 6,
) -> list[dict]:
    """
    Sinh danh sách fold walk-forward expanding window.

    Fold được sinh ngược từ TRAIN_END về quá khứ — fold cuối (fold n_folds)
    luôn có val period gần test thực nhất.
    Fold bị bỏ nếu train_end < TRAIN_START + 2 năm (không đủ data).

    Returns:
        list[dict]: mỗi dict có keys: fold_id, train_end, val_start, val_end.
    """
    global_train_end = pd.Timestamp(TRAIN_END)
    min_train_end    = pd.Timestamp(TRAIN_START) + pd.DateOffset(years=2)

    folds = []
    for i in range(n_folds):
        offset    = i * val_months
        val_end   = global_train_end - pd.DateOffset(months=offset)
        val_start = val_end - pd.DateOffset(months=val_months) + pd.DateOffset(days=1)
        train_end = val_start - pd.DateOffset(months=gap_months) - pd.DateOffset(days=1)

        if train_end < min_train_end:
            logger.warning(f"   Fold {n_folds - i} bỏ qua: train quá ngắn.")
            continue

        folds.append({
            'fold_id'  : n_folds - i,
            'train_end': train_end,
            'val_start': val_start,
            'val_end'  : val_end,
        })

    folds = sorted(folds, key=lambda x: x['fold_id'])
    logger.info(f"   {len(folds)} fold hợp lệ:")
    for f in folds:
        logger.info(
            f"     Fold {f['fold_id']}: "
            f"Train [{TRAIN_START} → {f['train_end'].date()}] | "
            f"Val   [{f['val_start'].date()} → {f['val_end'].date()}]"
        )
    return folds


def walk_forward_cv(
    model_factory             ,
    X_full                    : pd.DataFrame,
    y_full                    : pd.DataFrame,
    n_folds                   : int  = 4,
    val_months                : int  = 6,
    gap_months                : int  = 6,
    rebuild_seasonal_per_fold : bool = True,
) -> pd.DataFrame:
    """
    Walk-forward CV với expanding window.

    Mỗi fold:
        1. Tạo train/val mask từ cột 'Date'.
        2. (Nếu rebuild_seasonal_per_fold) Rebuild Nhóm 3 với train_mask của fold.
        3. Fit imputer + scaler trên X_train fold.
        4. Transform X_val.
        5. Train fresh model (model_factory()), predict, tính metrics.

    Tại sao rebuild_seasonal_per_fold:
        Seasonal stats của Nhóm 3 phải fit trên đúng train data của fold đó.
        Nếu không rebuild, fold 1 dùng seasonal stats từ toàn bộ 2012-2022
        → fold 1 "dễ" hơn thực tế → CV metrics lạc quan.

    Args:
        model_factory (callable): Hàm không tham số trả về model mới sklearn API.
                                  VD: lambda: LGBMRegressor(n_estimators=500)
        X_full (pd.DataFrame)  : Feature matrix đầy đủ (có cột 'Date').
        y_full (pd.DataFrame)  : Target vector (Date, Revenue, COGS).
        n_folds (int)          : Số fold.
        val_months (int)       : Độ dài val window (tháng).
        gap_months (int)       : Gap train_end → val_start (tháng).
        rebuild_seasonal_per_fold (bool): Rebuild Nhóm 3 mỗi fold.

    Returns:
        pd.DataFrame: cv_results với cột:
            fold_id, target, train_days, val_days, mae, rmse, r2, mape,
            train_end, val_start, val_end.
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU WALK-FORWARD CV")
    logger.info("=" * 60)

    check_leakage(X_full)

    feature_cols = [c for c in X_full.columns if c != 'Date']
    folds        = _build_cv_folds(X_full['Date'], n_folds, val_months, gap_months)

    master = None
    if rebuild_seasonal_per_fold:
        mp = config.PROCESSED / 'master_table.parquet'
        if mp.exists():
            master = pd.read_parquet(mp)
            master['Date'] = pd.to_datetime(master['Date'])
            master = master.sort_values('Date').reset_index(drop=True)
        else:
            logger.warning("   master_table.parquet không tìm thấy → tắt rebuild_seasonal.")
            rebuild_seasonal_per_fold = False

    all_results = []

    for fold in folds:
        fid, train_end, val_start, val_end = (
            fold['fold_id'], fold['train_end'], fold['val_start'], fold['val_end']
        )
        logger.info(f"\n--- Fold {fid} ---")

        train_mask = (
            (X_full['Date'] >= pd.Timestamp(TRAIN_START)) &
            (X_full['Date'] <= train_end)
        )
        val_mask = (
            (X_full['Date'] >= val_start) &
            (X_full['Date'] <= val_end)
        )

        if not train_mask.any() or not val_mask.any():
            logger.warning(f"   Fold {fid}: không đủ data, bỏ qua.")
            continue

        X_fold = X_full.copy()

        # Rebuild seasonal stats với đúng train mask của fold
        if rebuild_seasonal_per_fold and master is not None:
            fold_train_mask = (
                (master['Date'] >= pd.Timestamp(TRAIN_START)) &
                (master['Date'] <= train_end)
            )
            seasonal_fold = feature_eng.build_seasonal_stats(master, fold_train_mask)
            seasonal_cols = [c for c in seasonal_fold.columns if c != 'Date']
            X_fold = X_fold.drop(
                columns=[c for c in seasonal_cols if c in X_fold.columns],
                errors='ignore'
            ).merge(seasonal_fold, on='Date', how='left')
            logger.info(f"   Seasonal rebuilt: {fold_train_mask.sum()} ngày train")

        feat_cols = [c for c in X_fold.columns if c != 'Date']
        X_tr_raw  = X_fold.loc[train_mask, feat_cols].copy()
        X_vl_raw  = X_fold.loc[val_mask,   feat_cols].copy()
        y_tr      = y_full.loc[train_mask, TARGETS].copy()
        y_vl      = y_full.loc[val_mask,   TARGETS].copy()

        logger.info(f"   Train: {train_mask.sum()} ngày | Val: {val_mask.sum()} ngày")

        medians  = _fit_imputer(X_tr_raw)
        X_tr_imp = _apply_imputer(X_tr_raw, medians)
        X_vl_imp = _apply_imputer(X_vl_raw, medians)
        scaler   = _fit_scaler(X_tr_imp)
        X_tr_sc  = _apply_scaler(X_tr_imp, scaler)
        X_vl_sc  = _apply_scaler(X_vl_imp, scaler)

        # Sample weights cho fold train
        fold_dates   = y_full.loc[train_mask, 'Date']
        fold_weights = compute_sample_weights(fold_dates)

        for target in TARGETS:
            y_tr_t = y_tr[target].values
            y_vl_t = y_vl[target].values

            if np.isnan(y_tr_t).all() or np.isnan(y_vl_t).all():
                logger.warning(f"   Fold {fid} | {target}: toàn NaN, bỏ qua.")
                continue

            model = model_factory()
            # Truyền sample_weight nếu model hỗ trợ
            try:
                model.fit(X_tr_sc, y_tr_t, sample_weight=fold_weights)
            except TypeError:
                model.fit(X_tr_sc, y_tr_t)

            y_pred = model.predict(X_vl_sc)
            mae    = mean_absolute_error(y_vl_t, y_pred)
            rmse   = root_mean_squared_error(y_vl_t, y_pred)
            r2     = r2_score(y_vl_t, y_pred)
            denom  = np.where(np.abs(y_vl_t) < 1e-8, 1e-8, y_vl_t)
            mape   = np.mean(np.abs((y_vl_t - y_pred) / denom)) * 100

            logger.info(
                f"   Fold {fid} | {target:7s}: "
                f"MAE={mae:>12,.0f}  RMSE={rmse:>12,.0f}  "
                f"R²={r2:.4f}  MAPE={mape:.2f}%"
            )
            all_results.append({
                'fold_id'   : fid,
                'target'    : target,
                'train_days': int(train_mask.sum()),
                'val_days'  : int(val_mask.sum()),
                'mae'       : mae,
                'rmse'      : rmse,
                'r2'        : r2,
                'mape'      : mape,
                'train_end' : train_end,
                'val_start' : val_start,
                'val_end'   : val_end,
            })

    if not all_results:
        logger.error("Không có fold nào thành công.")
        return pd.DataFrame()

    cv_results = pd.DataFrame(all_results)

    logger.info("\n" + "=" * 60)
    logger.info("CV SUMMARY")
    logger.info("=" * 60)
    for target in TARGETS:
        tdf = cv_results[cv_results['target'] == target]
        if tdf.empty:
            continue
        stability = tdf['mae'].std() / tdf['mae'].mean() if tdf['mae'].mean() > 0 else 0
        logger.info(
            f"  {target:7s} | "
            f"MAE={tdf['mae'].mean():>12,.0f} ±{tdf['mae'].std():>10,.0f} | "
            f"R²={tdf['r2'].mean():.4f} ±{tdf['r2'].std():.4f} | "
            f"Stability={stability:.3f} {'✅' if stability < 0.2 else '⚠️'}"
        )

    cv_results.to_parquet(config.FEATURES / 'cv_results.parquet', index=False)
    return cv_results


# ==========================================
# 5. FINAL SPLIT
# ==========================================

def prepare_final_split(
    X_full : pd.DataFrame,
    y_full : pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tạo X_train / X_test / y_train cuối cùng cho train.py.

    Quy trình:
        1. Leakage check (safety net lần cuối).
        2. Filter cold start — drop 728 ngày đầu train (toàn NaN lag features).
        3. Tính + export sample_weights.parquet.
        4. Fit imputer trên X_train → transform X_train + X_test.
        5. Fit scaler trên X_train → transform X_train + X_test.
        6. Gắn cột 'Date' vào X_test (inference.py cần để ghép submission).
        7. Export tất cả parquet + joblib artifacts.

    Cold start (728 ngày đầu):
        Hầu hết YoY lag features NaN trong giai đoạn này vì chưa đủ lịch sử.
        Model học trên những ngày này = học pattern của imputed medians,
        không phải pattern thật → drop để tránh nhiễu.

    Args:
        X_full (pd.DataFrame): Feature matrix đầy đủ (có cột 'Date').
        y_full (pd.DataFrame): Target vector (Date, Revenue, COGS).

    Returns:
        X_train, X_test, y_train
    """
    logger.info("=" * 60)
    logger.info("CHUẨN BỊ FINAL SPLIT")
    logger.info("=" * 60)

    check_leakage(X_full)

    feature_cols     = [c for c in X_full.columns if c != 'Date']
    cold_start_end   = pd.Timestamp(TRAIN_START) + pd.DateOffset(days=728)

    train_mask = (
        (X_full['Date'] > cold_start_end) &          # bỏ cold start
        (X_full['Date'] <= pd.Timestamp(TRAIN_END))
    )
    test_mask = (
        (X_full['Date'] >= pd.Timestamp(TEST_START)) &
        (X_full['Date'] <= pd.Timestamp(TEST_END))
    )

    logger.info(
        f"   Train: {train_mask.sum()} ngày "
        f"({cold_start_end.date()} → {TRAIN_END}, cold start 728 ngày đã drop) | "
        f"Test: {test_mask.sum()} ngày"
    )

    y_train     = y_full.loc[train_mask].dropna(subset=TARGETS).copy()
    valid_idx   = y_train.index
    X_train_raw = X_full.loc[X_full.index.isin(valid_idx), feature_cols].copy()
    X_test_raw  = X_full.loc[test_mask, feature_cols].copy()

    # Sample weights
    weights = compute_sample_weights(y_train['Date'])
    pd.DataFrame({
        'Date'         : y_train['Date'].values,
        'sample_weight': weights,
    }).to_parquet(config.FEATURES / 'sample_weights.parquet', index=False)

    # Impute
    medians     = _fit_imputer(X_train_raw)
    X_train_imp = _apply_imputer(X_train_raw, medians)
    X_test_imp  = _apply_imputer(X_test_raw,  medians)

    # Scale
    scaler      = _fit_scaler(X_train_imp)
    X_train_sc  = _apply_scaler(X_train_imp, scaler)
    X_test_sc   = _apply_scaler(X_test_imp,  scaler)

    # Gắn Date vào X_test
    test_dates = X_full.loc[test_mask, ['Date']].reset_index(drop=True)
    X_test_sc  = pd.concat([test_dates, X_test_sc.reset_index(drop=True)], axis=1)

    # Export
    X_train_sc.to_parquet(config.FEATURES / 'X_train.parquet',  index=False)
    X_test_sc.to_parquet( config.FEATURES / 'X_test.parquet',   index=False)
    y_train.to_parquet(   config.FEATURES / 'y_train.parquet',  index=False)
    joblib.dump(scaler,   config.FEATURES / 'scaler.joblib')
    joblib.dump(medians,  config.FEATURES / 'train_medians.joblib')

    logger.info("=" * 60)
    logger.info("FINAL SPLIT HOÀN TẤT ✅")
    logger.info(f"  X_train        : {X_train_sc.shape}")
    logger.info(f"  X_test         : {X_test_sc.shape}")
    logger.info(f"  y_train        : {y_train.shape}")
    logger.info(f"  sample_weights : {weights.shape}")
    logger.info("=" * 60)

    return X_train_sc, X_test_sc, y_train


# ==========================================
# ENTRY POINT
# ==========================================

def run_validation(
    model_factory             = None,   # [FIX] default None — hợp lệ khi run_cv=False
    n_folds                   : int  = 4,
    val_months                : int  = 6,
    gap_months                : int  = 6,
    rebuild_seasonal_per_fold : bool = True,
    run_cv                    : bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Entry point của validation pipeline.

    Thứ tự:
        1. Load X_full + y_full (build nếu chưa có).
        2. (Nếu run_cv) walk_forward_cv().
        3. prepare_final_split().

    Args:
        model_factory (callable): Hàm trả về model mới. Bắt buộc khi run_cv=True.
                                  VD: lambda: LGBMRegressor(n_estimators=500)
        n_folds (int)           : Số fold CV.
        val_months (int)        : Độ dài val window.
        gap_months (int)        : Gap train → val.
        rebuild_seasonal_per_fold (bool): Rebuild seasonal stats mỗi fold.
        run_cv (bool)           : False = bỏ qua CV, chỉ chạy final split.

    Returns:
        (cv_results, X_train, X_test, y_train)
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU VALIDATION PIPELINE")
    logger.info("=" * 60)

    x_path = config.FEATURES / 'X_full.parquet'
    y_path = config.FEATURES / 'y_full.parquet'

    if x_path.exists() and y_path.exists():
        logger.info("Load X_full + y_full từ cache...")
        X_full = pd.read_parquet(x_path)
        y_full = pd.read_parquet(y_path)
        X_full['Date'] = pd.to_datetime(X_full['Date'])
        y_full['Date'] = pd.to_datetime(y_full['Date'])
    else:
        logger.info("Cache chưa có → chạy feature_eng.run_feature_engineering()...")
        X_full, y_full = feature_eng.run_feature_engineering()

    cv_results = pd.DataFrame()
    if run_cv:
        if model_factory is None:
            raise ValueError("model_factory không được None khi run_cv=True.")
        cv_results = walk_forward_cv(
            model_factory             = model_factory,
            X_full                    = X_full,
            y_full                    = y_full,
            n_folds                   = n_folds,
            val_months                = val_months,
            gap_months                = gap_months,
            rebuild_seasonal_per_fold = rebuild_seasonal_per_fold,
        )

    X_train, X_test, y_train = prepare_final_split(X_full, y_full)
    # [FIX] Bỏ duplicate export — prepare_final_split() đã lưu X_train + y_train rồi

    logger.info("=" * 60)
    logger.info("VALIDATION PIPELINE HOÀN TẤT ✅")
    logger.info("=" * 60)

    return cv_results, X_train, X_test, y_train

if __name__ == "__main__": 
    run_validation()