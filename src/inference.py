"""
================================================================================
MODULE: INFERENCE PIPELINE (DỰ ĐOÁN VÀ XUẤT SUBMISSION)
================================================================================

Mục đích:
    Nhận artifacts từ train.py (model_Revenue.joblib, model_COGS.joblib) và
    validation.py (X_test.parquet), chạy predict, xuất submission.csv đúng
    format đề bài và một bản diagnostics để kiểm tra sanity trước khi nộp.

    Module này KHÔNG transform lại data, KHÔNG fit thêm bất cứ thứ gì.
    Mọi imputation và scaling đã hoàn tất trong validation.py.
    inference.py chỉ làm đúng 3 việc:
        Bước 1 — Load model + X_test artifacts.
        Bước 2 — Predict từng target, clip về miền hợp lệ.
        Bước 3 — Assemble + validate + xuất submission.csv.

Thiết kế clip predictions:
    Revenue và COGS không thể âm về mặt nghiệp vụ.
    clip(lower=0) là hard constraint, không phải tuning choice.
    Dùng np.maximum thay vì pd.clip để rõ intent.

Post-prediction sanity checks:
    1. Không có NaN trong predictions.
    2. Mọi giá trị >= 0 (revenue/cost không âm).
    3. Revenue >= COGS (gross margin dương) — cảnh báo nếu vi phạm.
    4. Thứ tự ngày đúng với TEST_START → TEST_END.
    5. Số dòng khớp với date spine test.

Đầu ra:
    submissions/submission.csv        — File nộp bài chính thức
    submissions/submission_diag.csv   — Diagnostics: predictions + train stats cho sanity check
"""

import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

from src.config import config

logger = config.get_logger(__name__)

# ── Constants (từ config) ──────────────────────────────────────────────────────
TARGETS    = config.TARGETS
TEST_START = config.TEST_START
TEST_END   = config.TEST_END


# ==========================================
# BƯỚC 1: LOAD ARTIFACTS
# ==========================================

def _load_inference_artifacts() -> tuple[pd.DataFrame, dict]:
    """
    Đọc X_test và model đã fit từ disk.

    X_test.parquet đã chứa cột 'Date' (được gắn bởi validation.prepare_final_split).
    Model artifacts là {target}.joblib được train.py xuất ra.

    Returns:
        X_test   (pd.DataFrame): Feature matrix test (có cột 'Date').
        models   (dict[str, model]): Model đã fit theo target.

    Raises:
        FileNotFoundError: Nếu X_test hoặc bất kỳ model nào chưa tồn tại.
    """
    required = {'X_test': config.FEATURES / 'X_test.parquet'}
    required.update({
        f'model_{t}': config.MODELS / f'model_{t}.joblib'
        for t in TARGETS
    })

    missing = [k for k, p in required.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Chưa tìm thấy artifacts: {missing}.\n"
            f"Thứ tự chạy đúng: --feat → (validation) → --train → --infer."
        )

    X_test = pd.read_parquet(required['X_test'])
    X_test['Date'] = pd.to_datetime(X_test['Date'])

    models = {
        target: joblib.load(required[f'model_{target}'])
        for target in TARGETS
    }

    logger.info(f"   X_test : {X_test.shape} | Date range: "
                f"{X_test['Date'].min().date()} → {X_test['Date'].max().date()}")
    for target, model in models.items():
        logger.info(f"   model_{target} : {type(model).__name__} loaded ")

    return X_test, models


# ==========================================
# BƯỚC 2: PREDICT
# ==========================================

def _predict(
    X_test  : pd.DataFrame,
    models  : dict,
) -> pd.DataFrame:
    """
    Chạy predict cho từng target, clip về miền hợp lệ [0, ∞).

    Revenue và COGS là đại lượng tài chính — không thể âm về nghiệp vụ.
    clip(lower=0) là hard business constraint, áp dụng sau predict.

    Args:
        X_test (pd.DataFrame): Feature matrix đã scale, có cột 'Date'.
        models (dict[str, model]): Model đã fit.

    Returns:
        pd.DataFrame: Predictions với cột Date + từng target.
    """
    date_col    = X_test[['Date']].reset_index(drop=True)
    feature_cols = [c for c in X_test.columns if c != 'Date']
    X           = X_test[feature_cols].values

    preds = {'Date': date_col['Date']}

    for target in TARGETS:
        if target not in models:
            logger.warning(f"   [{target}] Không có model → skip.")
            continue

        raw_pred          = models[target].predict(X)
        clipped_pred      = np.maximum(raw_pred, 0.0)   # hard constraint: không âm
        n_clipped         = int((raw_pred < 0).sum())

        if n_clipped > 0:
            logger.warning(
                f"   [{target}] {n_clipped} predictions âm → clip về 0 "
                f"(min raw = {raw_pred.min():,.0f})."
            )
        else:
            logger.info(f"   [{target}] Không có prediction âm ")

        preds[target] = clipped_pred
        logger.info(
            f"   [{target}] mean={clipped_pred.mean():>12,.0f} | "
            f"min={clipped_pred.min():>12,.0f} | "
            f"max={clipped_pred.max():>12,.0f}"
        )

    return pd.DataFrame(preds)


# ==========================================
# BƯỚC 3: SANITY CHECKS
# ==========================================

def _run_sanity_checks(predictions: pd.DataFrame) -> bool:
    """
    Kiểm tra tính hợp lệ của predictions trước khi xuất submission.

    5 checks theo thứ tự nghiêm trọng:
        1. Không có NaN — crash nếu vi phạm (fatal).
        2. Mọi giá trị >= 0 — crash nếu vi phạm (fatal, không nên xảy ra sau clip).
        3. Số dòng khớp date spine test — crash nếu vi phạm (fatal).
        4. Date range đúng TEST_START → TEST_END — crash nếu vi phạm (fatal).
        5. Revenue >= COGS — warning nếu vi phạm (nghiệp vụ: gross margin > 0).

    Args:
        predictions (pd.DataFrame): DataFrame với cột Date + targets.

    Returns:
        bool: True nếu pass tất cả checks (kể cả có warning).

    Raises:
        ValueError: Nếu vi phạm check fatal (1–4).
    """
    logger.info("   Chạy sanity checks...")
    passed = True

    # Check 1: Không có NaN
    nan_counts = predictions[TARGETS].isna().sum()
    if nan_counts.any():
        raise ValueError(
            f"[FATAL] Predictions chứa NaN:\n{nan_counts[nan_counts > 0]}"
        )
    logger.info("   [1/5] No NaN ")

    # Check 2: Không âm (sau clip không nên xảy ra, nhưng defense in depth)
    neg_counts = (predictions[TARGETS] < 0).sum()
    if neg_counts.any():
        raise ValueError(
            f"[FATAL] Predictions âm sau clip — unexpected:\n{neg_counts[neg_counts > 0]}"
        )
    logger.info("   [2/5] All >= 0 ")

    # Check 3: Số dòng
    expected_start = pd.Timestamp(TEST_START)
    expected_end   = pd.Timestamp(TEST_END)
    expected_dates = pd.date_range(expected_start, expected_end, freq='D')
    if len(predictions) != len(expected_dates):
        raise ValueError(
            f"[FATAL] Số dòng predictions ({len(predictions)}) "
            f"!= expected date spine ({len(expected_dates)})."
        )
    logger.info(f"   [3/5] Row count = {len(predictions)} ")

    # Check 4: Date range
    actual_min = predictions['Date'].min()
    actual_max = predictions['Date'].max()
    if actual_min != expected_start or actual_max != expected_end:
        raise ValueError(
            f"[FATAL] Date range sai. "
            f"Expected: {expected_start.date()} → {expected_end.date()} | "
            f"Actual: {actual_min.date()} → {actual_max.date()}"
        )
    logger.info(f"   [4/5] Date range {TEST_START} → {TEST_END} ")

    # Check 5: Revenue >= COGS (gross margin dương)
    if 'Revenue' in predictions.columns and 'COGS' in predictions.columns:
        margin_violations = (predictions['Revenue'] < predictions['COGS']).sum()
        if margin_violations > 0:
            pct = margin_violations / len(predictions) * 100
            logger.warning(
                f"   [5/5] ⚠️  {margin_violations} ngày ({pct:.1f}%) có Revenue < COGS "
                f"(gross margin âm). Kiểm tra lại model hoặc clip strategy."
            )
            passed = False  # Warning, không crash
        else:
            logger.info("   [5/5] Revenue >= COGS mọi ngày ")

    return passed


# ==========================================
# ASSEMBLE SUBMISSION
# ==========================================

def _assemble_submission(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Format predictions thành submission.csv theo đúng yêu cầu đề bài.

    Yêu cầu từ đề bài (Đề_bài_phần_3.txt):
        - Cột: Date, Revenue, COGS.
        - Giữ đúng thứ tự dòng (chronological, không shuffle).
        - Revenue và COGS là số thực (float), không làm tròn tại đây
          để tránh mất precision — BTC tự làm tròn nếu cần.

    Args:
        predictions (pd.DataFrame): Kết quả từ _predict(), có cột Date + targets.

    Returns:
        pd.DataFrame: submission DataFrame đúng format.
    """
    submission = predictions[['Date'] + TARGETS].copy()
    submission = submission.sort_values('Date').reset_index(drop=True)
    submission['Date'] = submission['Date'].dt.strftime('%Y-%m-%d')
    return submission


def _build_diagnostics(
    predictions : pd.DataFrame,
    X_test      : pd.DataFrame,
) -> pd.DataFrame:
    """
    Tạo bảng diagnostics để kiểm tra sanity trước khi nộp bài.

    Bổ sung thêm vào predictions:
        - gross_margin_pct : (Revenue - COGS) / Revenue × 100 — check lợi nhuận gộp.
        - cogs_ratio       : COGS / Revenue — check cấu trúc chi phí.
        - month            : Tháng — tiện lọc seasonal patterns.
        - year             : Năm.

    Không dùng để nộp bài — chỉ để bro verify bằng mắt trước khi submit.

    Args:
        predictions (pd.DataFrame): DataFrame predictions gốc (Date kiểu Timestamp).
        X_test      (pd.DataFrame): X_test (có cột 'Date').

    Returns:
        pd.DataFrame: Bảng diagnostics.
    """
    diag = predictions.copy()
    diag['Date'] = pd.to_datetime(diag['Date'])

    if 'Revenue' in diag.columns and 'COGS' in diag.columns:
        diag['gross_margin_pct'] = (
            (diag['Revenue'] - diag['COGS']) / diag['Revenue'].clip(lower=1) * 100
        ).round(2)
        diag['cogs_ratio'] = (
            diag['COGS'] / diag['Revenue'].clip(lower=1)
        ).round(4)

    diag['month'] = diag['Date'].dt.month
    diag['year']  = diag['Date'].dt.year

    return diag


# ==========================================
# ORCHESTRATOR
# ==========================================

def run_inference(force_rerun: bool = False) -> pd.DataFrame:
    """
    Entry point chính — thực thi toàn bộ Inference Pipeline.

    - force_rerun=False (default): Bỏ qua nếu submission.csv đã tồn tại.
    - force_rerun=True            : Chạy lại dù submission đã có.

    Thứ tự:
        1. Kiểm tra artifact đã tồn tại chưa.
        2. Load X_test + models.
        3. Predict + clip.
        4. Sanity checks.
        5. Assemble submission + diagnostics.
        6. Lưu submission.csv + submission_diag.csv.

    Args:
        force_rerun (bool): Mặc định False để tránh overwrite vô ý.

    Returns:
        pd.DataFrame: submission DataFrame (Date, Revenue, COGS).

    Raises:
        FileNotFoundError: Artifacts chưa tồn tại.
        ValueError       : Sanity check fatal thất bại.
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU INFERENCE PIPELINE")
    logger.info("=" * 60)

    submission_path = config.SUBMISSIONS / 'submission.csv'
    config.SUBMISSIONS.mkdir(parents=True, exist_ok=True)

    if not force_rerun and submission_path.exists():
        logger.info(
            "submission.csv đã tồn tại. Bỏ qua để tránh overwrite "
            "(đặt force_rerun=True để chạy lại)."
        )
        return pd.read_csv(submission_path)

    # ── Bước 1: Load ──────────────────────────────────────────────────────────
    logger.info("\nBước 1 — Load artifacts...")
    X_test, models = _load_inference_artifacts()

    # ── Bước 2: Predict ───────────────────────────────────────────────────────
    logger.info("\nBước 2 — Predict...")
    predictions = _predict(X_test, models)

    # ── Bước 3: Sanity checks ─────────────────────────────────────────────────
    logger.info("\nBước 3 — Sanity checks...")
    _run_sanity_checks(predictions)

    # ── Bước 4: Assemble + Export ─────────────────────────────────────────────
    logger.info("\nBước 4 — Assemble và xuất submission...")
    submission = _assemble_submission(predictions)
    submission.to_csv(submission_path, index=False)
    logger.info(f"   Đã lưu: {submission_path}")

    diag = _build_diagnostics(predictions, X_test)
    diag_path = config.SUBMISSIONS / 'submission_diag.csv'
    diag.to_csv(diag_path, index=False)
    logger.info(f"   Đã lưu: {diag_path}")

    # ── Summary ───────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE PIPELINE HOÀN TẤT ")
    logger.info(f"  Rows     : {len(submission)}")
    logger.info(f"  Columns  : {list(submission.columns)}")
    for target in TARGETS:
        col = submission[target]
        logger.info(
            f"  {target:<10}: mean={col.mean():>12,.0f} | "
            f"min={col.min():>12,.0f} | max={col.max():>12,.0f}"
        )
    if 'gross_margin_pct' in diag.columns:
        logger.info(
            f"  GrossMargin: mean={diag['gross_margin_pct'].mean():.1f}% | "
            f"min={diag['gross_margin_pct'].min():.1f}% | "
            f"max={diag['gross_margin_pct'].max():.1f}%"
        )
    logger.info(f"  Output   : {config.SUBMISSIONS}")
    logger.info("=" * 60)

    return submission