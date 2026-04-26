"""
================================================================================
MODULE: FEATURE ENGINEERING PIPELINE
================================================================================

Mục đích:
    Biến đổi master_table.parquet thành feature matrix X_full và target vector
    y_full theo toàn bộ date spine. Module này KHÔNG split, KHÔNG scale,
    KHÔNG impute — toàn bộ trách nhiệm đó thuộc về validation.py.

Chiến lược: Safe Lag Only (không recursive forecast)
    Câu hỏi kiểm tra với mỗi feature:
        "Ngày 2024-06-30, feature này có tính được từ thông tin đã biết không?"

    4 nhóm feature:
        Nhóm 1 — Calendar Features  : Chỉ cần date, an toàn tuyệt đối.
        Nhóm 2 — YoY Lag Features   : Lag >= 364 ngày + rolling volatility.
        Nhóm 3 — Seasonal Stats     : Fit trên train_mask, map theo calendar key.
        Nhóm 4 — Promo Calendar     : Extrapolate pattern từ promotions.csv.

    Tại sao lag_728 thay vì lag_365:
        Test kết thúc 2024-07-01.
        lag_365(2024-07-01) = 2023-07-01 → ngoài train → NaN toàn bộ H2/2024.
        lag_728(2024-07-01) = 2022-07-01 → trong train ✅

    Tại sao có thêm rolling volatility (Nhóm 2):
        Model không chỉ cần biết "cùng kỳ năm ngoái revenue bao nhiêu" mà còn
        cần biết "cùng kỳ năm ngoái có ổn định không". std cao = giai đoạn bất
        ổn (COVID-like), model nên conservative hơn khi dùng lag đó làm anchor.
        signal_quality = 1/(1+CV) tổng hợp thành 1 con số [0,1] đo độ tin cậy.

    Tại sao Nhóm 4 hợp lệ (không phải external data):
        promotions.csv cho thấy pattern lặp lại đều đặn mỗi năm:
          Spring Sale (tháng 3), Mid-Year Sale (tháng 6), Fall Launch (tháng 8),
          Year-End Sale (tháng 11), Urban Blowout & Rural Special (năm lẻ).
        Extrapolate sang 2023-2024 là suy luận từ internal data.

    Tại sao KHÔNG có regime flags:
        One-hot flag "ecom_boom / covid_period" chỉ xuất hiện 1 lần trong lịch sử.
        Tree model không thể học từ 1 lần xuất hiện → feature vô nghĩa thống kê.
        Thay vào đó: yoy_growth và rolling volatility đã encode "triệu chứng" của
        từng regime mà không cần đặt nhãn.

Đầu ra — data/features/:
    calendar_features.parquet       — Nhóm 1
    yoy_features.parquet            — Nhóm 2 (lag + volatility)
    seasonal_stats.parquet          — Nhóm 3
    promo_calendar_features.parquet — Nhóm 4
    X_full.parquet                  — Feature matrix toàn date spine (chưa split/scale)
    y_full.parquet                  — Target vector toàn date spine
    feature_metadata.parquet        — Tên, nhóm, null_pct_raw của từng feature
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from src.config import config

logger = config.get_logger(__name__)

# ── Constants (từ config) ──────────────────────────────────────────────────────
TRAIN_START     = config.TRAIN_START
TRAIN_END       = config.TRAIN_END
TEST_START      = config.TEST_START
TEST_END        = config.TEST_END
TARGETS         = config.TARGETS
SAFE_LAG_DAYS   = config.SAFE_LAG_DAYS
ROLLING_WINDOWS = config.ROLLING_WINDOWS


# ==========================================
# NHÓM 1: CALENDAR FEATURES
# ==========================================

def build_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """
    Temporal features thuần từ date — an toàn tuyệt đối với mọi horizon.

    Features:
        Basic      : year, quarter, month, week_of_year, day_of_month,
                     day_of_week (0=Mon), day_of_year
        Flags      : is_weekend, is_month_start/end, is_quarter_start/end,
                     is_year_start/end
        Cyclical   : sin/cos của month, day_of_week, week_of_year, day_of_year
                     — tránh model coi tháng 12→1 là "xa nhau"
        Trend      : years_since_start — capture organic business growth

    Args:
        dates (pd.Series): Series kiểu datetime.

    Returns:
        pd.DataFrame: Calendar features với cột 'Date'.
    """
    df = pd.DataFrame({'Date': dates})
    dt = df['Date'].dt

    df['year']         = dt.year
    df['quarter']      = dt.quarter
    df['month']        = dt.month
    df['week_of_year'] = dt.isocalendar().week.astype(int)
    df['day_of_month'] = dt.day
    df['day_of_week']  = dt.dayofweek
    df['day_of_year']  = dt.dayofyear

    df['is_weekend']       = (dt.dayofweek >= 5).astype(int)
    df['is_month_start']   = dt.is_month_start.astype(int)
    df['is_month_end']     = dt.is_month_end.astype(int)
    df['is_quarter_start'] = dt.is_quarter_start.astype(int)
    df['is_quarter_end']   = dt.is_quarter_end.astype(int)
    df['is_year_start']    = dt.is_year_start.astype(int)
    df['is_year_end']      = dt.is_year_end.astype(int)

    df['month_sin'] = np.sin(2 * np.pi * dt.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * dt.month / 12)
    df['dow_sin']   = np.sin(2 * np.pi * dt.dayofweek / 7)
    df['dow_cos']   = np.cos(2 * np.pi * dt.dayofweek / 7)
    df['week_sin']  = np.sin(2 * np.pi * dt.isocalendar().week.astype(int) / 52)
    df['week_cos']  = np.cos(2 * np.pi * dt.isocalendar().week.astype(int) / 52)
    df['doy_sin']   = np.sin(2 * np.pi * dt.dayofyear / 365)
    df['doy_cos']   = np.cos(2 * np.pi * dt.dayofyear / 365)

    base_year = pd.Timestamp(TRAIN_START).year
    df['years_since_start'] = (dt.year - base_year) + (dt.dayofyear - 1) / 365.0

    logger.info(f"   [Nhóm 1] Calendar: {df.shape[1] - 1} features")
    return df


# ==========================================
# NHÓM 2: YOY LAG + ROLLING VOLATILITY
# ==========================================

def build_yoy_features(master: pd.DataFrame) -> pd.DataFrame:
    """
    YoY lag, rolling mean, rolling volatility và signal quality features.

    An toàn với horizon 18 tháng:
        lag_728 an toàn tuyệt đối cho toàn bộ test period.
        lag_364/365/366 bị NaN ở H2/2024 → được impute bằng lag_728 × (1 + yoy_growth).
        Imputation này dùng thông tin đã biết nên an toàn, không phải train-fit imputation.

    Các cột được lag (12 signals):
        Revenue, COGS                           — targets chính
        daily_gross_revenue, daily_net_revenue  — transaction volume proxy
        daily_orders, daily_items_sold          — demand signals
        daily_returns, daily_refund_amount      — quality/churn signals
        total_sessions, total_page_views        — traffic leading indicators
        avg_daily_rating                        — customer sentiment
        daily_actual_discount                   — promotional pressure

    Với mỗi signal, các features được tạo:
        {alias}_lag{N}             : Raw lag tại N ngày trước (N trong SAFE_LAG_DAYS)
        {alias}_roll{W}_lag364     : Rolling mean W ngày, anchor tại lag_364
        {alias}_yoy_growth         : (lag_364 - lag_728) / lag_728, clip [-2, 10]
        {alias}_trend_slope        : (roll30 - roll90) / roll90, clip [-5, 5]

    Chỉ với Revenue và COGS (thêm volatility):
        rev/cogs_vol{W}_lag364     : Rolling std W ngày quanh cùng kỳ năm ngoái
        rev/cogs_cv{W}_lag364      : Coefficient of variation = std/mean, clip [0,10]
        rev/cogs_signal_quality{W} : 1/(1+CV) — 1.0=ổn định, ~0=bất ổn

        Ý nghĩa: COVID làm std cao → signal_quality thấp → model tự biết
        "thông tin cùng kỳ năm ngoái kém tin cậy, đừng tin quá nhiều vào lag_728".

    Args:
        master (pd.DataFrame): Master table đầy đủ.

    Returns:
        pd.DataFrame: YoY features với cột 'Date'.
    """
    df = master.copy().sort_values('Date').reset_index(drop=True)

    lag_targets = {
        'Revenue'             : 'rev',
        'COGS'                : 'cogs',
        'daily_gross_revenue' : 'gross_rev',
        'daily_net_revenue'   : 'net_rev',
        'daily_orders'        : 'orders',
        'daily_items_sold'    : 'items',
        'daily_returns'       : 'returns',
        'daily_refund_amount' : 'refund',
        'total_sessions'      : 'sessions',
        'total_page_views'    : 'pageviews',
        'avg_daily_rating'    : 'rating',
        'daily_actual_discount': 'discount',
    }
    volatility_aliases = {'rev', 'cogs'}

    result = df[['Date']].copy()

    for col, alias in lag_targets.items():
        if col not in df.columns:
            logger.warning(f"   '{col}' không có trong master_table, bỏ qua.")
            continue

        series        = df[col]
        series_lag364 = series.shift(364)

        # Raw lags
        for lag in SAFE_LAG_DAYS:
            result[f'{alias}_lag{lag}'] = series.shift(lag)

        # Rolling mean trên cùng kỳ năm ngoái
        for w in ROLLING_WINDOWS:
            result[f'{alias}_roll{w}_lag364'] = (
                series_lag364
                .rolling(w, min_periods=max(1, w // 2))
                .mean()
            )

        # YoY growth rate
        lag364 = series.shift(364)
        lag728 = series.shift(728)
        result[f'{alias}_yoy_growth'] = (
            (lag364 - lag728) / lag728.replace(0, np.nan)
        ).clip(-2, 10)

        # Trend slope: momentum ngắn hạn so với dài hạn của cùng kỳ
        roll30 = series_lag364.rolling(30, min_periods=10).mean()
        roll90 = series_lag364.rolling(90, min_periods=30).mean()
        result[f'{alias}_trend_slope'] = (
            (roll30 - roll90) / roll90.replace(0, np.nan)
        ).clip(-5, 5)

        # Rolling volatility — chỉ cho Revenue và COGS
        if alias in volatility_aliases:
            for w in (30, 90):
                vol  = series_lag364.rolling(w, min_periods=w // 2).std()
                mean = series_lag364.rolling(w, min_periods=w // 2).mean().replace(0, np.nan)
                cv   = (vol / mean.abs()).clip(0, 10)

                result[f'{alias}_vol{w}_lag364']    = vol
                result[f'{alias}_cv{w}_lag364']     = cv
                result[f'{alias}_signal_quality{w}'] = 1.0 / (1.0 + cv)

    # Impute lag_365/366 NaN ở H2/2024 bằng lag_728 × (1 + yoy_growth)
    # An toàn: chỉ dùng thông tin đã biết, không phải train-fit imputation
    for col, alias in [('Revenue', 'rev'), ('COGS', 'cogs')]:
        for lag in [365, 366]:
            lag_col = f'{alias}_lag{lag}'
            if lag_col not in result.columns:
                continue
            mask = result[lag_col].isna()
            if mask.sum() == 0:
                continue
            growth = result[f'{alias}_yoy_growth'].fillna(0)
            result.loc[mask, lag_col] = (
                result.loc[mask, f'{alias}_lag728'] * (1 + growth[mask])
            )
            logger.info(f"   Imputed {mask.sum()} NaN trong {lag_col} bằng lag728×(1+yoy_growth)")

    logger.info(f"   [Nhóm 2] YoY Lag + Volatility: {result.shape[1] - 1} features")
    return result


# ==========================================
# NHÓM 3: SEASONAL STATS
# ==========================================

def build_seasonal_stats(master: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """
    Thống kê mùa vụ lịch sử — fit ONLY trên train_mask, map theo calendar key.

    train_mask được truyền từ ngoài để hàm này tái sử dụng được trong cả:
        - run_feature_engineering() : dùng full train mask (2012→2022)
        - walk_forward_cv()         : dùng train mask của từng fold

    Aggregation keys:
        month, quarter, day_of_week, week_of_year, (month × day_of_week)

    Statistics per key (Revenue và COGS):
        mean, median, std, q25, q75, seasonal_idx
        seasonal_idx = group_mean / annual_mean — 1.0 là trung bình năm

    Args:
        master (pd.DataFrame): Master table đầy đủ.
        train_mask (pd.Series): Boolean mask chỉ train partition.

    Returns:
        pd.DataFrame: Seasonal stats features với cột 'Date'.
    """
    df       = master.copy().sort_values('Date').reset_index(drop=True)
    train_df = df[train_mask].copy()

    for key in ['month', 'quarter', 'day_of_week', 'week_of_year']:
        train_df[key] = getattr(train_df['Date'].dt, key if key != 'week_of_year' else 'isocalendar')
    train_df['month']        = train_df['Date'].dt.month
    train_df['quarter']      = train_df['Date'].dt.quarter
    train_df['day_of_week']  = train_df['Date'].dt.dayofweek
    train_df['week_of_year'] = train_df['Date'].dt.isocalendar().week.astype(int)

    result = df[['Date']].copy()
    result['month']        = df['Date'].dt.month
    result['quarter']      = df['Date'].dt.quarter
    result['day_of_week']  = df['Date'].dt.dayofweek
    result['week_of_year'] = df['Date'].dt.isocalendar().week.astype(int)

    annual_means = {
        t: train_df[t].mean() if t in train_df.columns else None
        for t in TARGETS
    }

    def _add_stats(keys, prefix):
        keys = [keys] if isinstance(keys, str) else list(keys)
        for target, tname in zip(TARGETS, ['rev', 'cogs']):
            if target not in train_df.columns:
                continue
            stats = (
                train_df.groupby(keys, observed=True)[target]
                .agg(
                    _mean  ='mean',
                    _median='median',
                    _std   ='std',
                    _q25   =lambda x: x.quantile(0.25),
                    _q75   =lambda x: x.quantile(0.75),
                )
                .reset_index()
            )
            stats['_std'] = stats['_std'].fillna(0)
            ann = annual_means[target]
            stats['_sidx'] = stats['_mean'] / ann if ann and ann > 0 else 1.0
            stats = stats.rename(columns={
                '_mean'  : f'{prefix}_{tname}_mean',
                '_median': f'{prefix}_{tname}_median',
                '_std'   : f'{prefix}_{tname}_std',
                '_q25'   : f'{prefix}_{tname}_q25',
                '_q75'   : f'{prefix}_{tname}_q75',
                '_sidx'  : f'{prefix}_{tname}_seasonal_idx',
            })
            new_cols = [c for c in stats.columns if c not in keys]
            merged   = result[keys].merge(stats, on=keys, how='left')
            for c in new_cols:
                result[c] = merged[c].values

    _add_stats('month',                  'by_month')
    _add_stats('quarter',                'by_quarter')
    _add_stats('day_of_week',            'by_dow')
    _add_stats('week_of_year',           'by_week')
    _add_stats(['month', 'day_of_week'], 'by_month_dow')

    result = result.drop(columns=['month', 'quarter', 'day_of_week', 'week_of_year'])

    logger.info(
        f"   [Nhóm 3] Seasonal Stats: {result.shape[1] - 1} features "
        f"(fit trên {train_mask.sum()} ngày train)"
    )
    return result


# ==========================================
# NHÓM 4: PROMO CALENDAR FEATURES
# ==========================================

def build_promo_calendar_features(dates: pd.Series) -> pd.DataFrame:
    """
    Extrapolate lịch khuyến mãi từ pattern promotions.csv sang toàn bộ date range.

    Pattern phát hiện từ 50 promo records (2013–2022) — lặp lại đều đặn:
        Spring Sale   : 18/3 → 17/4,  percentage 12%,  mỗi năm
        Mid-Year Sale : 23/6 → 22/7,  percentage 18%,  mỗi năm
        Fall Launch   : 30/8 → 01/10, percentage 10%,  mỗi năm
        Year-End Sale : 18/11 → 31/12, percentage 20%, mỗi năm
        Urban Blowout : 30/7 → 02/9,  fixed 50,        năm lẻ (Streetwear)
        Rural Special : 30/1 → 01/3,  percentage 15%,  năm lẻ (Outdoor)

    Features:
        is_promo_active          : Có ít nhất 1 promo đang chạy
        n_active_promos          : Số promo đồng thời
        max_discount_pct_active  : % discount cao nhất đang active
        has_{spring/mid_year/fall/year_end/streetwear/outdoor}_promo : flags
        days_to_next_promo       : Số ngày đến promo kế tiếp, clip [0, 180]
        days_since_last_promo    : Số ngày kể từ promo cuối, clip [0, 180]

    Args:
        dates (pd.Series): Series kiểu datetime.

    Returns:
        pd.DataFrame: Promo calendar features với cột 'Date'.
    """
    min_year = dates.dt.year.min()
    max_year = dates.dt.year.max()

    records = []
    for year in range(min_year, max_year + 1):
        records += [
            {'start': f'{year}-03-18', 'end': f'{year}-04-17', 'disc_pct': 12.0, 'flag': 'spring'},
            {'start': f'{year}-06-23', 'end': f'{year}-07-22', 'disc_pct': 18.0, 'flag': 'mid_year'},
            {'start': f'{year}-08-30', 'end': f'{year}-10-01', 'disc_pct': 10.0, 'flag': 'fall'},
            {'start': f'{year}-11-18', 'end': f'{year}-12-31', 'disc_pct': 20.0, 'flag': 'year_end'},
        ]
        if year % 2 == 1:
            records += [
                {'start': f'{year}-07-30', 'end': f'{year}-09-02', 'disc_pct': 0.0,  'flag': 'streetwear'},
                {'start': f'{year}-01-30', 'end': f'{year}-03-01', 'disc_pct': 15.0, 'flag': 'outdoor'},
            ]

    promos = pd.DataFrame(records)
    promos['start'] = pd.to_datetime(promos['start'])
    promos['end']   = pd.to_datetime(promos['end'])

    result = pd.DataFrame({'Date': dates.values}).sort_values('Date').reset_index(drop=True)
    flags  = ['spring', 'mid_year', 'fall', 'year_end', 'streetwear', 'outdoor']

    result['is_promo_active']         = 0
    result['n_active_promos']         = 0
    result['max_discount_pct_active'] = 0.0
    for f in flags:
        result[f'has_{f}_promo'] = 0

    for _, p in promos.iterrows():
        mask = (result['Date'] >= p['start']) & (result['Date'] <= p['end'])
        result.loc[mask, 'is_promo_active']          = 1
        result.loc[mask, 'n_active_promos']         += 1
        result.loc[mask, 'max_discount_pct_active']  = np.maximum(
            result.loc[mask, 'max_discount_pct_active'], p['disc_pct']
        )
        result.loc[mask, f"has_{p['flag']}_promo"]  = 1

    all_starts = sorted(promos['start'].tolist())
    all_ends   = sorted(promos['end'].tolist())

    result['days_to_next_promo'] = result['Date'].apply(
        lambda d: min([(s - d).days for s in all_starts if s > d], default=999)
    ).clip(0, 180)
    result['days_since_last_promo'] = result['Date'].apply(
        lambda d: min([(d - e).days for e in all_ends if e < d], default=999)
    ).clip(0, 180)

    logger.info(f"   [Nhóm 4] Promo Calendar: {result.shape[1] - 1} features")
    return result


# ==========================================
# MERGE & EXPORT
# ==========================================

def _merge_all_features(
    master   : pd.DataFrame,
    calendar : pd.DataFrame,
    yoy      : pd.DataFrame,
    seasonal : pd.DataFrame,
    promo    : pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Merge 4 nhóm feature vào một DataFrame và tách target.

    Không split, không impute, không scale — trách nhiệm của validation.py.
    Cột 'Date' được giữ trong X_full để validation.py tạo time-based mask.

    Returns:
        X_full (pd.DataFrame): Feature matrix toàn date spine (có cột 'Date').
        y_full (pd.DataFrame): Target vector (Date, Revenue, COGS).
    """
    merged = (
        master[['Date'] + TARGETS]
        .merge(calendar, on='Date', how='left')
        .merge(yoy,      on='Date', how='left')
        .merge(seasonal, on='Date', how='left')
        .merge(promo,    on='Date', how='left')
    )

    feature_cols = [c for c in merged.columns if c not in ['Date'] + TARGETS]

    def _get_group(c):
        if c in calendar.columns: return 'calendar'
        if c in yoy.columns:      return 'yoy'
        if c in promo.columns:    return 'promo'
        return 'seasonal'

    pd.DataFrame({
        'feature_name' : feature_cols,
        'group'        : [_get_group(c) for c in feature_cols],
        'null_pct_raw' : [round(merged[c].isna().mean() * 100, 2) for c in feature_cols],
    }).to_parquet(config.FEATURES / 'feature_metadata.parquet', index=False)

    X_full = merged[['Date'] + feature_cols].copy()
    y_full = merged[['Date'] + TARGETS].copy()

    logger.info(f"   X_full: {X_full.shape} | y_full: {y_full.shape}")
    logger.info(f"   NaN raw (trước imputation): {X_full[feature_cols].isna().sum().sum()}")
    return X_full, y_full


# ==========================================
# ENTRY POINT
# ==========================================

def run_feature_engineering() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pipeline chính: Load → Build 4 nhóm → Merge → Export X_full, y_full.

    Returns:
        (X_full, y_full) — chưa split, chưa impute, chưa scale.
        Truyền thẳng vào validation.run_validation().

    Raises:
        FileNotFoundError: master_table.parquet chưa tồn tại.
    """
    logger.info("=" * 60)
    logger.info("BẮT ĐẦU FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)

    master_path = config.PROCESSED / 'master_table.parquet'
    if not master_path.exists():
        raise FileNotFoundError(
            f"{master_path} không tồn tại. Chạy data_prep.run_prep() trước."
        )
    config.FEATURES.mkdir(parents=True, exist_ok=True)

    logger.info("1. Load master_table...")
    master = pd.read_parquet(master_path)
    master['Date'] = pd.to_datetime(master['Date'])
    master = master.sort_values('Date').reset_index(drop=True)

    train_mask = (
        (master['Date'] >= pd.Timestamp(TRAIN_START)) &
        (master['Date'] <= pd.Timestamp(TRAIN_END))
    )
    logger.info(
        f"   Train: {train_mask.sum()} ngày ({TRAIN_START} → {TRAIN_END}) | "
        f"Full spine: {len(master)} ngày"
    )

    logger.info("2. Build Calendar Features (Nhóm 1)...")
    calendar = build_calendar_features(master['Date'])
    calendar.to_parquet(config.FEATURES / 'calendar_features.parquet', index=False)

    logger.info("3. Build YoY Lag + Volatility Features (Nhóm 2)...")
    yoy = build_yoy_features(master)
    yoy.to_parquet(config.FEATURES / 'yoy_features.parquet', index=False)

    logger.info("4. Build Seasonal Stats (Nhóm 3)...")
    seasonal = build_seasonal_stats(master, train_mask)
    seasonal.to_parquet(config.FEATURES / 'seasonal_stats.parquet', index=False)

    logger.info("5. Build Promo Calendar Features (Nhóm 4)...")
    promo = build_promo_calendar_features(master['Date'])
    promo.to_parquet(config.FEATURES / 'promo_calendar_features.parquet', index=False)

    logger.info("6. Merge & Export...")
    X_full, y_full = _merge_all_features(master, calendar, yoy, seasonal, promo)
    X_full.to_parquet(config.FEATURES / 'X_full.parquet', index=False)
    y_full.to_parquet(config.FEATURES / 'y_full.parquet', index=False)

    logger.info("=" * 60)
    logger.info("FEATURE ENGINEERING HOÀN TẤT ✅")
    logger.info(f"  X_full : {X_full.shape} | y_full : {y_full.shape}")
    logger.info(f"  Output : {config.FEATURES}")
    logger.info("=" * 60)

    return X_full, y_full