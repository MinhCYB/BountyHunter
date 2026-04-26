"""
================================================================================
MODULE: MODEL SELECTION (SO SÁNH + GRID SEARCH HYPERPARAMETER TUNING)
================================================================================

Mục đích:
    (A) So sánh nhiều model architecture trên Walk-Forward CV.
    (B) Grid Search hyperparams cho từng target độc lập (Revenue ≠ COGS).
    (C) Promote best params → models/best_params.parquet để train.py đọc
        trực tiếp — artifact rõ ràng, persist cross-session, có thể diff.

Kiến trúc 3 tầng:

    Tầng 1 — Config-driven (config.py):
        GRID_SEARCH_SPACE  : Dict định nghĩa param grid cho từng model class.
        CV_SETTINGS        : n_folds, val_months, gap_months.
        MODELS             : Output dir — best_params.parquet được lưu tại đây.

    Tầng 2 — Grid Search per target (module này):
        tune_hyperparams() : Với mỗi target × param combo, chạy Walk-Forward CV,
                             tổng hợp MAE_mean, chọn best combo, ghi vào config.

    Tầng 3 — Model comparison (module này, giữ nguyên từ v1):
        compare_models()   : So sánh N model architecture trên cùng CV setting.

Tại sao tune riêng từng target:
    Revenue nhạy với promo, seasonality mạnh → cần num_leaves cao hơn để
    capture non-linear interactions. COGS ổn định hơn, đơn giản hơn → model
    shallower tránh overfit vào noise ngắn hạn. Tune chung = dùng params
    của target khó hơn cho target dễ → suboptimal cho cả hai.

Tại sao Walk-Forward CV thay vì holdout:
    Holdout cố định overfit vào một period. WF-CV đánh giá stability theo
    thời gian — quan trọng hơn với time series có seasonality mạnh như đây.
    Khi BTC hỏi "tại sao chọn params này", bro có CV stability score để trả lời.

Workflow:
    Bước 1: Thêm/sửa GRID_SEARCH_SPACE trong config.py.
    Bước 2: python -m src.model_selection --tune   ← grid search
    Bước 3: Đọc tuning_results.parquet, kiểm tra best params trong log.
    Bước 4: Nếu đồng ý → python -m src.model_selection --apply  ← ghi parquet
    Bước 5: python main.py --train  ← train.py đọc models/best_params.parquet

    Hoặc chỉ so sánh architecture:
    python -m src.model_selection --compare

Đầu ra — data/features/:
    tuning_results.parquet   — Chi tiết mọi param combo × fold × target
    tuning_summary.parquet   — Best params per target + CV scores
    comparison_results.parquet
    comparison_summary.parquet

[QUAN TRỌNG] Module này KHÔNG tự động apply params vào train.py.
    Cần dùng --apply (hoặc gọi apply_best_params()) để promote ra parquet.
    Đây là checkpoint bắt buộc — bro review trước khi train.
"""

import argparse
import itertools
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

from src.config import config
from src import feature_eng, validation

logger = config.get_logger(__name__)

# ── Constants (từ config) ──────────────────────────────────────────────────────
TARGETS = config.TARGETS


# ==========================================
# TẦNG 1 HELPERS — ĐỌC CONFIG
# ==========================================

def _get_cv_settings() -> dict:
    """
    Đọc CV settings từ config.CV_SETTINGS.

    Fallback về default nếu config chưa khai báo CV_SETTINGS — đảm bảo
    backward compatible với config cũ.

    Returns:
        dict với keys: n_folds, val_months, gap_months, rebuild_seasonal_per_fold.
    """
    defaults = {
        'n_folds'                   : 4,
        'val_months'                : 6,
        'gap_months'                : 6,
        'rebuild_seasonal_per_fold' : True,
    }
    user_settings = getattr(config, 'CV_SETTINGS', {})
    return {**defaults, **user_settings}


def _get_grid_search_space() -> dict:
    """
    Đọc param grid từ config.GRID_SEARCH_SPACE.

    Format mong đợi trong config.py:
        GRID_SEARCH_SPACE = {
            'LGBMRegressor': {
                'n_estimators'    : [500, 800, 1000],
                'learning_rate'   : [0.03, 0.05],
                'num_leaves'      : [63, 127],
                'min_child_samples': [10, 20],
                'subsample'       : [0.8],
                'colsample_bytree': [0.8],
                'reg_alpha'       : [0.0, 0.1],
                'reg_lambda'      : [0.0, 0.1],
            }
        }

    Trả về empty dict nếu chưa khai báo — tune_hyperparams() sẽ báo lỗi rõ ràng.

    Returns:
        dict[str, dict[str, list]]: Param grid theo model class name.
    """
    space = getattr(config, 'GRID_SEARCH_SPACE', {})
    if not space:
        logger.warning(
            "config.GRID_SEARCH_SPACE chưa được khai báo.\n"
            "Thêm vào config.py:\n"
            "    GRID_SEARCH_SPACE = {\n"
            "        'LGBMRegressor': {\n"
            "            'n_estimators': [500, 1000],\n"
            "            'learning_rate': [0.03, 0.05],\n"
            "            ...\n"
            "        }\n"
            "    }"
        )
    return space


def _load_feature_artifacts() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load X_full + y_full từ cache, hoặc chạy feature engineering nếu chưa có.

    Dùng chung cho cả tune_hyperparams() và compare_models() để tránh
    load data 2 lần khi chạy cả --tune và --compare trong cùng session.

    Returns:
        (X_full, y_full) với cột 'Date' đã là pd.Timestamp.
    """
    x_path = config.FEATURES / 'X_full.parquet'
    y_path = config.FEATURES / 'y_full.parquet'

    if x_path.exists() and y_path.exists():
        logger.info("Load X_full + y_full từ cache...")
        X_full = pd.read_parquet(x_path)
        y_full = pd.read_parquet(y_path)
    else:
        logger.info("Cache chưa có → chạy feature_eng.run_feature_engineering()...")
        X_full, y_full = feature_eng.run_feature_engineering()

    X_full['Date'] = pd.to_datetime(X_full['Date'])
    y_full['Date'] = pd.to_datetime(y_full['Date'])
    return X_full, y_full


# ==========================================
# TẦNG 2 — GRID SEARCH PER TARGET
# ==========================================

def _expand_param_grid(param_grid: dict) -> list[dict]:
    """
    Expand param grid dict thành list các combo hyperparams.

    Tương đương sklearn.model_selection.ParameterGrid nhưng không cần import
    sklearn riêng — và transparent hơn về số lượng combo.

    Args:
        param_grid (dict[str, list]): VD {'n_estimators': [500, 1000], 'lr': [0.03, 0.05]}

    Returns:
        list[dict]: Mỗi phần tử là một bộ params hoàn chỉnh.
        VD: [{'n_estimators': 500, 'lr': 0.03}, {'n_estimators': 500, 'lr': 0.05}, ...]
    """
    keys   = list(param_grid.keys())
    values = list(param_grid.values())
    return [dict(zip(keys, combo)) for combo in itertools.product(*values)]


def _make_factory(model_class, params: dict, random_state: int):
    """
    Tạo callable factory cho model với params cụ thể.

    Dùng default argument capture (params=params) để tránh closure bug —
    nếu dùng lambda: factory, params sẽ bị capture by reference, tất cả
    factory trong vòng lặp sẽ dùng params cuối cùng.

    Args:
        model_class : Class model (VD: LGBMRegressor).
        params (dict): Hyperparams cần truyền vào constructor.
        random_state (int): Từ config.RANDOM_STATE.

    Returns:
        callable: Không tham số, trả về model instance mới khi gọi.
    """
    def factory(cls=model_class, p=params, rs=random_state):
        try:
            return cls(**p, random_state=rs, n_jobs=-1, verbosity=-1)
        except TypeError:
            # Fallback: một số model không nhận verbosity hoặc n_jobs
            try:
                return cls(**p, random_state=rs, n_jobs=-1)
            except TypeError:
                return cls(**p, random_state=rs)
    return factory


def tune_hyperparams(
    model_class_name : str | None = None,
    target_filter    : list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Grid Search hyperparams trên Walk-Forward CV, tune riêng từng target.

    Quy trình cho mỗi target × param combo:
        1. Tạo model factory với combo params đó.
        2. Chạy validation.walk_forward_cv() — cùng logic CV với pipeline chính.
        3. Tổng hợp MAE_mean (metric chính), RMSE_mean, R²_mean, MAE_std (stability).
        4. Chọn combo có MAE_mean thấp nhất và MAE_std thấp nhất (tie-break).

    Tại sao MAE_mean là metric chính (không phải R²):
        R² phụ thuộc variance của target — một model đơn giản predict mean cũng
        có R² cao nếu target biến động ít. MAE trực quan hơn: "sai trung bình
        bao nhiêu đồng" — BTC dễ hiểu, bro dễ giải thích.

    Tại sao MAE_std là tie-break:
        Trong datathon, model ổn định qua các fold quan trọng hơn model có
        một fold đặc biệt tốt. MAE_std thấp = model reliable, không bị
        overfit vào một period cụ thể.

    Args:
        model_class_name (str | None):
            Tên model class cần tune (VD: 'LGBMRegressor').
            None = tune tất cả class trong GRID_SEARCH_SPACE.
        target_filter (list[str] | None):
            Chỉ tune cho các target này. None = tune tất cả TARGETS.

    Returns:
        tuning_results (pd.DataFrame):
            Chi tiết mọi param combo × fold × target.
            Schema: target, combo_id, params_json, fold_id, mae, rmse, r2, mape,
                    train_days, val_days.

        tuning_summary (pd.DataFrame):
            Best params per target. Schema: target, model_class, best_params_json,
            mae_mean, mae_std, rmse_mean, r2_mean, n_folds_ok, is_best.

    Raises:
        ValueError: Nếu GRID_SEARCH_SPACE rỗng hoặc model_class_name không tìm thấy.
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU GRID SEARCH HYPERPARAMETER TUNING")
    logger.info("=" * 60)

    grid_space = _get_grid_search_space()
    if not grid_space:
        raise ValueError(
            "config.GRID_SEARCH_SPACE rỗng. Xem docstring _get_grid_search_space() "
            "để biết format khai báo."
        )

    # Filter model class nếu được chỉ định
    if model_class_name is not None:
        if model_class_name not in grid_space:
            raise ValueError(
                f"'{model_class_name}' không có trong GRID_SEARCH_SPACE. "
                f"Có: {list(grid_space.keys())}"
            )
        grid_space = {model_class_name: grid_space[model_class_name]}

    targets_to_tune = target_filter if target_filter else TARGETS
    cv_settings     = _get_cv_settings()

    # Load data một lần
    X_full, y_full = _load_feature_artifacts()

    # Import model classes động theo tên trong config
    model_registry = _build_model_registry()

    all_tune_rows    = []
    best_per_target  = {}   # target → {'model_class', 'params', 'mae_mean', 'mae_std', ...}

    for class_name, param_grid in grid_space.items():
        if class_name not in model_registry:
            logger.warning(
                f"   [{class_name}] Không import được — thư viện chưa cài? Bỏ qua."
            )
            continue

        model_class = model_registry[class_name]
        combos      = _expand_param_grid(param_grid)

        logger.info(f"\nModel class : {class_name}")
        logger.info(f"Param grid  : {len(combos)} combos × {len(targets_to_tune)} targets "
                    f"× {cv_settings['n_folds']} folds "
                    f"= {len(combos) * len(targets_to_tune) * cv_settings['n_folds']} CV runs")

        for target in targets_to_tune:
            logger.info(f"\n  [Target: {target}] Tuning {len(combos)} param combos...")

            target_combo_scores = []   # (combo_id, mae_mean, mae_std, rmse_mean, r2_mean, fold_rows)

            for combo_id, params in enumerate(combos):
                factory = _make_factory(model_class, params, config.RANDOM_STATE)
                params_json = json.dumps(params, sort_keys=True)

                try:
                    # CV chỉ cho 1 target — filter y_full để giảm noise log
                    fold_results = validation.walk_forward_cv(
                        model_factory             = factory,
                        X_full                    = X_full,
                        y_full                    = y_full,
                        n_folds                   = cv_settings['n_folds'],
                        val_months                = cv_settings['val_months'],
                        gap_months                = cv_settings['gap_months'],
                        rebuild_seasonal_per_fold = cv_settings['rebuild_seasonal_per_fold'],
                    )

                    if fold_results.empty:
                        continue

                    # Filter chỉ lấy kết quả của target đang tune
                    target_rows = fold_results[fold_results['target'] == target].copy()
                    if target_rows.empty:
                        continue

                    mae_mean  = target_rows['mae'].mean()
                    mae_std   = target_rows['mae'].std()
                    rmse_mean = target_rows['rmse'].mean()
                    r2_mean   = target_rows['r2'].mean()
                    n_folds   = len(target_rows)

                    target_combo_scores.append((combo_id, mae_mean, mae_std, rmse_mean, r2_mean, n_folds))

                    # Ghi detail rows
                    for _, fold_row in target_rows.iterrows():
                        all_tune_rows.append({
                            'target'      : target,
                            'model_class' : class_name,
                            'combo_id'    : combo_id,
                            'params_json' : params_json,
                            'fold_id'     : fold_row.get('fold_id'),
                            'mae'         : fold_row.get('mae'),
                            'rmse'        : fold_row.get('rmse'),
                            'r2'          : fold_row.get('r2'),
                            'mape'        : fold_row.get('mape'),
                            'train_days'  : fold_row.get('train_days'),
                            'val_days'    : fold_row.get('val_days'),
                        })

                    logger.info(
                        f"    combo {combo_id:>3} | MAE={mae_mean:>12,.0f} ±{mae_std:>10,.0f} "
                        f"| R²={r2_mean:.4f} | params={params}"
                    )

                except Exception as exc:
                    logger.error(f"    combo {combo_id} lỗi: {exc}")

            if not target_combo_scores:
                logger.warning(f"  [{target}] Không có combo nào thành công.")
                continue

            # Chọn best: primary = MAE_mean thấp nhất, tie-break = MAE_std thấp nhất
            target_combo_scores.sort(key=lambda x: (x[1], x[2]))
            best_combo_id, best_mae_mean, best_mae_std, best_rmse_mean, best_r2_mean, best_n_folds = target_combo_scores[0]
            best_params = combos[best_combo_id]

            logger.info(
                f"\n  ✅ [{target}] Best combo #{best_combo_id}: {best_params}\n"
                f"     MAE_mean={best_mae_mean:,.0f} | MAE_std={best_mae_std:,.0f} "
                f"| R²={best_r2_mean:.4f}"
            )

            # Giữ best per target (update nếu class này tốt hơn class trước)
            if target not in best_per_target or best_mae_mean < best_per_target[target]['mae_mean']:
                best_per_target[target] = {
                    'target'          : target,
                    'model_class'     : class_name,
                    'best_params_json': json.dumps(best_params, sort_keys=True),
                    'best_params'     : best_params,
                    'mae_mean'        : best_mae_mean,
                    'mae_std'         : best_mae_std,
                    'rmse_mean'       : best_rmse_mean,
                    'r2_mean'         : best_r2_mean,
                    'n_folds_ok'      : best_n_folds,
                    'combo_id'        : best_combo_id,
                }

    # ── Assemble outputs ──────────────────────────────────────────────────────
    tuning_results  = pd.DataFrame(all_tune_rows) if all_tune_rows else pd.DataFrame()
    summary_rows    = []

    for target, info in best_per_target.items():
        summary_rows.append({
            'target'          : info['target'],
            'model_class'     : info['model_class'],
            'best_params_json': info['best_params_json'],
            'mae_mean'        : info['mae_mean'],
            'mae_std'         : info['mae_std'],
            'rmse_mean'       : info['rmse_mean'],
            'r2_mean'         : info['r2_mean'],
            'n_folds_ok'      : info['n_folds_ok'],
        })

    tuning_summary = pd.DataFrame(summary_rows)

    # Export
    config.FEATURES.mkdir(parents=True, exist_ok=True)
    if not tuning_results.empty:
        tuning_results.to_parquet(config.FEATURES / 'tuning_results.parquet', index=False)
    tuning_summary.to_parquet(config.FEATURES / 'tuning_summary.parquet', index=False)

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("GRID SEARCH HOÀN TẤT")
    logger.info("=" * 60)
    for _, row in tuning_summary.iterrows():
        params = json.loads(row['best_params_json'])
        logger.info(
            f"  [{row['target']:<10}] {row['model_class']:<20} "
            f"MAE={row['mae_mean']:>12,.0f} ±{row['mae_std']:>10,.0f} "
            f"R²={row['r2_mean']:.4f}"
        )
        logger.info(f"             Best params: {params}")

    logger.info("\n  [NEXT] Xem kết quả, nếu đồng ý chạy:")
    logger.info("         python -m src.model_selection --apply")
    logger.info("         Sau đó: python main.py --train")
    logger.info("=" * 60)

    return tuning_results, tuning_summary


# ==========================================
# TẦNG 2 — PROMOTE BEST PARAMS → models/best_params.parquet
# ==========================================

def apply_best_params(dry_run: bool = False) -> pd.DataFrame:
    """
    Promote best params từ tuning_summary.parquet → models/best_params.parquet.

    Bước này là checkpoint bắt buộc — bro phải gọi tường minh sau khi review
    tuning_summary.parquet. Không tự động chạy sau tune_hyperparams().

    Tại sao lưu ra parquet thay vì ghi vào config:
        - Parquet là artifact tường minh: có thể diff, kiểm tra lại bất cứ lúc nào.
        - Không phụ thuộc Python runtime — restart session vẫn hoạt động bình thường.
        - train.py đọc parquet trực tiếp, không cần biết config.TUNED_PARAMS là gì.
        - Dễ audit: biết chính xác params nào được dùng để train, khi nào promote.

    Schema best_params.parquet (1 dòng per target):
        target       : str   — 'Revenue' | 'COGS'
        model_class  : str   — 'LGBMRegressor' | 'XGBRegressor' | ...
        params_json  : str   — JSON string của best hyperparams
        mae_mean     : float
        mae_std      : float
        rmse_mean    : float
        r2_mean      : float
        n_folds_ok   : int
        promoted_at  : str   — timestamp ISO 8601 lúc promote

    dry_run=True:
        Log params sẽ được promote mà KHÔNG ghi file.
        Dùng để review trước khi confirm.

    Args:
        dry_run (bool): True = chỉ log, không ghi. False = ghi thật ra parquet.

    Returns:
        pd.DataFrame: best_params DataFrame đã promote.

    Raises:
        FileNotFoundError: tuning_summary.parquet chưa tồn tại.
    """
    from datetime import datetime, timezone

    summary_path = config.FEATURES / 'tuning_summary.parquet'
    if not summary_path.exists():
        raise FileNotFoundError(
            "tuning_summary.parquet chưa tồn tại.\n"
            "Chạy tune_hyperparams() trước: python -m src.model_selection --tune"
        )

    summary = pd.read_parquet(summary_path)

    logger.info("=" * 60)
    logger.info(f"{'[DRY RUN] ' if dry_run else ''}PROMOTE BEST PARAMS → best_params.parquet")
    logger.info("=" * 60)

    rows = []
    for _, row in summary.iterrows():
        target      = row['target']
        model_class = row['model_class']
        params      = json.loads(row['best_params_json'])

        logger.info(
            f"  [{target:<10}] {model_class:<20} "
            f"MAE={row['mae_mean']:>12,.0f} ±{row['mae_std']:>10,.0f} "
            f"R²={row['r2_mean']:.4f}"
        )
        logger.info(f"             Params : {params}")

        rows.append({
            'target'      : target,
            'model_class' : model_class,
            'params_json' : row['best_params_json'],
            'mae_mean'    : row['mae_mean'],
            'mae_std'     : row['mae_std'],
            'rmse_mean'   : row['rmse_mean'],
            'r2_mean'     : row['r2_mean'],
            'n_folds_ok'  : row['n_folds_ok'],
            'promoted_at' : datetime.now(timezone.utc).isoformat(),
        })

    best_params_df = pd.DataFrame(rows)

    if not dry_run:
        out_path = config.MODELS / 'best_params.parquet'
        config.MODELS.mkdir(parents=True, exist_ok=True)
        best_params_df.to_parquet(out_path, index=False)
        logger.info(f"\n  ✅ Đã lưu: {out_path}")
        logger.info("  [NEXT] python main.py --train")
    else:
        logger.info("\n  [DRY RUN] Không ghi. Bỏ --dry-run để promote thật.")

    logger.info("=" * 60)
    return best_params_df


# ==========================================
# TẦNG 3 — SO SÁNH MODEL ARCHITECTURE
# ==========================================

def compare_models(
    candidates                : dict,
    n_folds                   : int  | None = None,
    val_months                : int  | None = None,
    gap_months                : int  | None = None,
    rebuild_seasonal_per_fold : bool | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Chạy Walk-Forward CV cho từng model candidate, tổng hợp bảng so sánh.

    CV settings ưu tiên theo thứ tự:
        1. Tham số truyền trực tiếp vào hàm (nếu không None).
        2. config.CV_SETTINGS (nếu khai báo).
        3. Default hardcode (n_folds=4, val_months=6, gap_months=6).

    Args:
        candidates (dict[str, callable]):
            Key   = tên model (VD: "LightGBM_baseline").
            Value = callable không tham số → model instance mới.
        n_folds, val_months, gap_months, rebuild_seasonal_per_fold:
            Override CV settings. None = dùng config hoặc default.

    Returns:
        (comparison_results, comparison_summary) — đã lưu ra parquet.
    """
    cv = _get_cv_settings()
    # Override nếu truyền tường minh
    if n_folds                   is not None: cv['n_folds']                   = n_folds
    if val_months                is not None: cv['val_months']                = val_months
    if gap_months                is not None: cv['gap_months']                = gap_months
    if rebuild_seasonal_per_fold is not None: cv['rebuild_seasonal_per_fold'] = rebuild_seasonal_per_fold

    logger.info("=" * 60)
    logger.info(f"BẮT ĐẦU SO SÁNH {len(candidates)} MODEL CANDIDATES")
    logger.info(f"CV Settings: {cv}")
    logger.info("=" * 60)

    X_full, y_full = _load_feature_artifacts()
    all_results    = []

    for name, factory in candidates.items():
        logger.info("-" * 60)
        logger.info(f"Đang đánh giá: [{name}]")
        logger.info("-" * 60)
        try:
            fold_results = validation.walk_forward_cv(
                model_factory             = factory,
                X_full                    = X_full,
                y_full                    = y_full,
                n_folds                   = cv['n_folds'],
                val_months                = cv['val_months'],
                gap_months                = cv['gap_months'],
                rebuild_seasonal_per_fold = cv['rebuild_seasonal_per_fold'],
            )
            if fold_results.empty:
                logger.warning(f"   [{name}] Không có fold nào thành công — bỏ qua.")
                continue

            fold_results.insert(0, 'candidate_name', name)
            all_results.append(fold_results)
            logger.info(f"   [{name}] Hoàn tất {len(fold_results)} fold-target rows.")

        except Exception as exc:
            logger.error(f"   [{name}] Lỗi khi chạy CV: {exc}")

    if not all_results:
        logger.error("Không có candidate nào thành công.")
        return pd.DataFrame(), pd.DataFrame()

    comparison_results = pd.concat(all_results, ignore_index=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    summary_rows = []
    for cname in comparison_results['candidate_name'].unique():
        for target in TARGETS:
            subset = comparison_results[
                (comparison_results['candidate_name'] == cname) &
                (comparison_results['target'] == target)
            ]
            if subset.empty:
                continue
            summary_rows.append({
                'candidate_name': cname,
                'target'        : target,
                'mae_mean'      : subset['mae'].mean(),
                'mae_std'       : subset['mae'].std(),
                'rmse_mean'     : subset['rmse'].mean(),
                'rmse_std'      : subset['rmse'].std(),
                'r2_mean'       : subset['r2'].mean(),
                'r2_std'        : subset['r2'].std(),
                'mape_mean'     : subset['mape'].mean() if 'mape' in subset.columns else np.nan,
                'mape_std'      : subset['mape'].std()  if 'mape' in subset.columns else np.nan,
                'n_folds_ok'    : len(subset),
            })

    comparison_summary = pd.DataFrame(summary_rows)

    config.FEATURES.mkdir(parents=True, exist_ok=True)
    comparison_results.to_parquet(config.FEATURES / 'comparison_results.parquet', index=False)
    comparison_summary.to_parquet(config.FEATURES / 'comparison_summary.parquet', index=False)

    # ── Print ─────────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("KẾT QUẢ SO SÁNH")
    logger.info("=" * 60)
    for target in TARGETS:
        logger.info(f"\n  TARGET: {target}")
        logger.info(f"  {'Candidate':<30} {'MAE_mean':>14} {'MAE_std':>12} {'R²_mean':>10} {'R²_std':>9}")
        logger.info(f"  {'-'*30} {'-'*14} {'-'*12} {'-'*10} {'-'*9}")
        tdf = comparison_summary[comparison_summary['target'] == target].sort_values('mae_mean')
        for _, row in tdf.iterrows():
            logger.info(
                f"  {row['candidate_name']:<30} "
                f"{row['mae_mean']:>14,.0f} {row['mae_std']:>12,.0f} "
                f"{row['r2_mean']:>10.4f} {row['r2_std']:>9.4f}"
            )
    logger.info("=" * 60)

    return comparison_results, comparison_summary


# ==========================================
# HELPERS — MODEL REGISTRY
# ==========================================

def _build_model_registry() -> dict:
    """
    Import các model class được hỗ trợ và trả về dict ánh xạ tên → class.

    Thêm model mới tại đây khi muốn mở rộng GRID_SEARCH_SPACE.

    Returns:
        dict[str, type]: VD {'LGBMRegressor': LGBMRegressor, ...}
    """
    registry = {}

    try:
        from lightgbm import LGBMRegressor
        registry['LGBMRegressor'] = LGBMRegressor
    except ImportError:
        logger.warning("lightgbm chưa cài — 'LGBMRegressor' không khả dụng.")

    try:
        from xgboost import XGBRegressor
        registry['XGBRegressor'] = XGBRegressor
    except ImportError:
        logger.warning("xgboost chưa cài — 'XGBRegressor' không khả dụng.")

    try:
        from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
        from sklearn.linear_model import Ridge
        registry['RandomForestRegressor']    = RandomForestRegressor
        registry['GradientBoostingRegressor']= GradientBoostingRegressor
        registry['Ridge']                    = Ridge
    except ImportError:
        logger.warning("scikit-learn chưa cài — sklearn models không khả dụng.")

    return registry


def _build_default_candidates() -> dict:
    """
    Bộ model candidate mặc định cho compare_models().

    Nếu models/best_params.parquet tồn tại (sau khi --apply), tự động
    dùng tuned params cho LGBM_baseline. Nếu chưa có, dùng reasonable defaults.

    Returns:
        dict[str, callable]: Candidates sẵn sàng truyền vào compare_models().
    """
    registry = _build_model_registry()
    tuned    = {}
    rs       = config.RANDOM_STATE
    candidates = {}

    try:
        LGBMRegressor = registry['LGBMRegressor']

        # Nếu đã tune → dùng tuned params, else → baseline
        lgbm_params = tuned.get('Revenue', {}).get('params', {
            'n_estimators': 500, 'learning_rate': 0.05, 'num_leaves': 63,
        })
        candidates['LGBM_baseline'] = _make_factory(LGBMRegressor, lgbm_params, rs)

        candidates['LGBM_deep'] = _make_factory(LGBMRegressor, {
            'n_estimators': 1000, 'learning_rate': 0.03, 'num_leaves': 127,
            'min_child_samples': 20, 'subsample': 0.8, 'colsample_bytree': 0.8,
            'reg_alpha': 0.1, 'reg_lambda': 0.1,
        }, rs)
    except KeyError:
        pass

    try:
        XGBRegressor = registry['XGBRegressor']
        candidates['XGB_baseline'] = _make_factory(XGBRegressor, {
            'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
        }, rs)
    except KeyError:
        pass

    try:
        RF = registry['RandomForestRegressor']
        candidates['RF_baseline'] = _make_factory(RF, {
            'n_estimators': 200, 'max_depth': None,
        }, rs)
    except KeyError:
        pass

    if not candidates:
        raise ImportError(
            "Không có thư viện ML nào. Cài: pip install lightgbm xgboost scikit-learn"
        )

    logger.info(f"Loaded {len(candidates)} default candidates: {list(candidates.keys())}")
    return candidates


# ==========================================
# ENTRY POINTS
# ==========================================

def run_tune(model_class_name: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry point cho --tune. Chạy grid search trên config.GRID_SEARCH_SPACE."""
    return tune_hyperparams(model_class_name=model_class_name)


def run_apply(dry_run: bool = False) -> dict:
    """Entry point cho --apply. Promote best params ra models/best_params.parquet."""
    return apply_best_params(dry_run=dry_run)


def run_compare(candidates: dict | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Entry point cho --compare. So sánh model architecture."""
    if candidates is None:
        candidates = _build_default_candidates()
    return compare_models(candidates)


def run_model_selection(
    candidates : dict | None = None,
    n_folds    : int  | None = None,
    val_months : int  | None = None,
    gap_months : int  | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Entry point backward-compatible — gọi từ notebook hoặc code cũ.
    Chỉ chạy compare_models() với candidates truyền vào hoặc default.
    """
    if candidates is None:
        candidates = _build_default_candidates()
    return compare_models(
        candidates = candidates,
        n_folds    = n_folds,
        val_months = val_months,
        gap_months = gap_months,
    )


# ==========================================
# CLI 
# ==========================================

def add_arguments(parser: argparse.ArgumentParser) -> None:
    """
    Đăng ký các argument của model_selection vào parser của main.py.

    Thiết kế để gọi từ main.py:
        from src import model_selection
        model_selection.add_arguments(parser)

    Tất cả argument được đặt trong nhóm "Model Selection" riêng biệt
    để --help của main.py hiển thị gọn và có cấu trúc.

    Args:
        parser (argparse.ArgumentParser): Parser của main.py để gắn thêm args.
    """
    group = parser.add_argument_group(
        "Model Selection",
        description="Thực nghiệm — đứng ngoài --all pipeline. Chạy độc lập khi cần."
    )
    group.add_argument(
        "--tune",
        action="store_true",
        help="Grid Search hyperparams theo config.GRID_SEARCH_SPACE (tune riêng Revenue + COGS)",
    )
    group.add_argument(
        "--apply",
        action="store_true",
        help="Promote best params từ tuning_summary.parquet → models/best_params.parquet",
    )
    group.add_argument(
        "--dry-run",
        action="store_true",
        help="Dùng với --apply: preview params sẽ được promote mà không ghi file",
    )
    group.add_argument(
        "--compare",
        action="store_true",
        help="So sánh nhiều model architecture trên Walk-Forward CV",
    )
    group.add_argument(
        "--model",
        type=str,
        default=None,
        metavar="CLASS",
        help="Tên model class cần tune (VD: LGBMRegressor). Mặc định: tất cả trong GRID_SEARCH_SPACE",
    )


def dispatch(args: argparse.Namespace) -> None:
    """
    Thực thi các lệnh model_selection dựa trên args đã parse.

    Được gọi từ main.py sau khi parser.parse_args(). Kiểm tra từng flag
    và gọi entry point tương ứng theo đúng thứ tự logic:
        tune → apply → compare
    (tune phải chạy trước apply; apply độc lập với compare).

    Args:
        args (argparse.Namespace): Kết quả từ parser.parse_args() của main.py.

    Returns:
        bool: True nếu có ít nhất một lệnh model_selection được thực thi.
              main.py dùng giá trị này để quyết định có log "Done" không.
    """
    executed = False

    if getattr(args, 'tune', False):
        run_tune(model_class_name=getattr(args, 'model', None))
        executed = True

    if getattr(args, 'apply', False):
        run_apply(dry_run=getattr(args, 'dry_run', False))
        executed = True

    if getattr(args, 'compare', False):
        run_compare()
        executed = True

    return executed


if __name__ == "__main__":
    # Standalone mode: python -m src.model_selection [--tune|--apply|--compare]
    # Dùng khi không muốn đi qua main.py, hoặc để test độc lập.
    _parser = argparse.ArgumentParser(
        description="Model Selection — BountyHunter Datathon 2026",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Ví dụ:\n"
            "  python -m src.model_selection --tune\n"
            "  python -m src.model_selection --tune --model LGBMRegressor\n"
            "  python -m src.model_selection --apply --dry-run\n"
            "  python -m src.model_selection --apply\n"
            "  python -m src.model_selection --compare\n"
            "\n"
            "Hoặc chạy qua main.py (khuyến nghị):\n"
            "  python main.py --tune\n"
            "  python main.py --apply --dry-run\n"
        ),
    )
    add_arguments(_parser)
    _args = _parser.parse_args()

    if not any([
        getattr(_args, 'tune', False),
        getattr(_args, 'apply', False),
        getattr(_args, 'compare', False),
    ]):
        _parser.print_help()
    else:
        dispatch(_args)