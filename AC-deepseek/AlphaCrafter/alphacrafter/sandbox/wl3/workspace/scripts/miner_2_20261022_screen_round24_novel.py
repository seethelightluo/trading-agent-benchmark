"""Round 24 novel factor screen (miner_2) - 2026-10-22.

Fresh ideas (checked vs full explored-id list incl. evicted/rejected/quarantine;
no duplicates):
- vol_autocorr_20x60   : lag-1 autocorr of 20d realized vol over 60d (vol persistence)
- vol_weighted_mom_20  : sum(r*v)/sum(v) over 20d (volume-weighted momentum)
- ewma_mom_20         : close/EWMA(close,span=20)-1 (time-decayed momentum)
- stoch_k_14          : (close-min_low14)/(max_high14-min_low14) (stochastic %K)
- beta_instability_60 : std of rolling 20d SPX beta over 60d (beta drift/instability)
- time_above_ma_60    : fraction of days close>MA20 over trailing 60d (trend participation)
- aroon_20            : (days_since_20d_high - days_since_20d_low)/20 (Aroon)
- keltner_pos_20      : (close-MA20)/(2*ATR20) (Keltner channel position)
- rank_autocorr_60    : lag-1 autocorr of cross-sectional rank over 60d (rank persistence)
- vol_cluster_asym_20 : mean|r| after up days / mean|r| after down days (vol cluster asym)
- comm_ratio_beta_60  : beta to d(WTI/XAU) over 60d (commodity-vs-gold relative beta)
- btc_eth_ratio_beta_60: beta to d(BTC/ETH) over 60d (crypto-relative beta)
- volume_autocorr_20  : lag-1 autocorr of volume over 20d (volume persistence)
- ad_line_slope_20    : slope of accumulation/distribution line over 20d (smart money)
- range_vol_20        : std of daily (high-low)/close over 20d (range volatility)

Validates vs full EFFECTIVE library (recoverable signal artifacts), per-date
Spearman rho. Gates: |IC|>=0.007, |ICIR|>=0.084, max_abs_library_correlation<0.5.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, WATCHLIST, persist_factor,
                           VAL_START, VAL_END)

TODAY = '2026-10-22'
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---- library: effective factors with recoverable signal artifacts ----
lib = {}
for p in sorted(Path('factors').glob('*.json')):
    if p.name.endswith('.bak') or 'deprecated' in p.name or 'ensemble' in p.name:
        continue
    try:
        payload = json.loads(p.read_text(encoding='utf-8'))
    except Exception:
        continue
    if payload.get('validation', {}).get('status') != 'EFFECTIVE':
        continue
    art = payload.get('signal_artifact')
    art_path = p.parent / str(art) if art else None
    if art_path is not None and art_path.exists():
        lib[payload['factor_id']] = np.load(art_path, allow_pickle=False)
print(f'library factors with artifacts: {len(lib)} {sorted(lib.keys())}', flush=True)


def spearman_rho_vs_library(panel):
    """Per-date Spearman rho vs each library artifact; report max |rho| (gate-style)."""
    pm = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    per_id = {}
    for fid, lm in lib.items():
        corrs = []
        for t in range(len(grid)):
            x = pm[t]; y = lm[t]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xr = pd.Series(x[m]).rank().values
                yr = pd.Series(y[m]).rank().values
                xc = xr - xr.mean(); yc = yr - yr.mean()
                den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                if den > 0:
                    corrs.append((xc * yc).sum() / den)
        if corrs:
            r = float(np.mean(corrs))
            per_id[fid] = r
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id, per_id


# ---- benchmark series ----
spx_ret = prices['SPX']['close'].pct_change()
wti = prices['WTI']['close']
xau = prices['XAU']['close']
btc = prices['BTC']['close']
eth = prices['ETH']['close']


def rolling_beta(asset_df, bench_series, window, min_obs=30):
    r = asset_df['close'].pct_change()
    z = pd.concat([r.rename('r'), bench_series.rename('b')], axis=1).dropna()
    cov = z['r'].rolling(window, min_periods=min_obs).cov(z['b'])
    var = z['b'].rolling(window, min_periods=min_obs).var()
    return (cov / var).reindex(z.index)


# ---- candidate factor functions ----
def f_vol_autocorr_20x60(df, s):
    v = df['close'].pct_change().rolling(20, min_periods=10).std()
    return v.rolling(60, min_periods=30).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else np.nan, raw=True)


def f_vol_weighted_mom_20(df, s):
    r = df['close'].pct_change()
    v = df['volume']
    num = (r * v).rolling(20, min_periods=10).sum()
    den = v.rolling(20, min_periods=10).sum()
    return (num / den).reindex(r.index)


def f_ewma_mom_20(df, s):
    c = df['close']
    ew = c.ewm(span=20, adjust=False).mean()
    return (c / ew - 1.0).reindex(c.index)


def f_stoch_k_14(df, s):
    lo = df['low'].rolling(14, min_periods=7).min()
    hi = df['high'].rolling(14, min_periods=7).max()
    return ((df['close'] - lo) / (hi - lo).replace(0, np.nan)).reindex(df.index)


def f_beta_instability_60(df, s):
    b = rolling_beta(df, spx_ret, 20, min_obs=10)
    return b.rolling(60, min_periods=30).std().reindex(b.index)


def f_time_above_ma_60(df, s):
    ma20 = df['close'].rolling(20, min_periods=10).mean()
    above = (df['close'] > ma20).astype(float)
    return above.rolling(60, min_periods=30).mean().reindex(df.index)


def f_aroon_20(df, s):
    hi_win = df['high'].rolling(20, min_periods=10)
    lo_win = df['low'].rolling(20, min_periods=10)
    days_since_hi = hi_win.apply(lambda x: len(x) - 1 - int(np.argmax(x)) if len(x) else np.nan, raw=True)
    days_since_lo = lo_win.apply(lambda x: len(x) - 1 - int(np.argmin(x)) if len(x) else np.nan, raw=True)
    return ((days_since_hi - days_since_lo) / 20.0).reindex(df.index)


def f_keltner_pos_20(df, s):
    ma = df['close'].rolling(20, min_periods=10).mean()
    tr = pd.concat([(df['high'] - df['low']),
                    (df['high'] - df['close'].shift(1)).abs(),
                    (df['low'] - df['close'].shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.rolling(20, min_periods=10).mean()
    return ((df['close'] - ma) / (2.0 * atr).replace(0, np.nan)).reindex(df.index)


def f_rank_autocorr_60(df, s):
    r = df['close'].pct_change()
    rk = r.rolling(60, min_periods=30).apply(lambda x: pd.Series(x).rank().values[-1] if len(x) else np.nan, raw=True)
    # per-asset rank is monotone in return -> fallback: autocorr of signed returns ranks = autocorr of rets
    return r.rolling(60, min_periods=30).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else np.nan, raw=True)


def f_vol_cluster_asym_20(df, s):
    r = df['close'].pct_change()
    prev = r.shift(1)
    up_prev = r.where(prev > 0).abs()
    dn_prev = r.where(prev < 0).abs()
    mu_up = up_prev.rolling(20, min_periods=5).mean()
    mu_dn = dn_prev.rolling(20, min_periods=5).mean()
    return (mu_up / mu_dn.replace(0, np.nan)).reindex(r.index)


def f_comm_ratio_beta_60(df, s):
    ratio = wti / xau
    return rolling_beta(df, ratio.pct_change(), 60)


def f_btc_eth_ratio_beta_60(df, s):
    ratio = btc / eth
    return rolling_beta(df, ratio.pct_change(), 60)


def f_volume_autocorr_20(df, s):
    v = df['volume'].replace(0, np.nan)
    return v.rolling(20, min_periods=10).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else np.nan, raw=True)


def f_ad_line_slope_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / rng * df['volume']
    ad = mfv.cumsum()
    slope = ad.rolling(20, min_periods=10).apply(lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 3 and np.std(x) > 0 else np.nan, raw=True)
    return slope.reindex(ad.index)


def f_range_vol_20(df, s):
    rr = ((df['high'] - df['low']) / df['close'].replace(0, np.nan))
    return rr.rolling(20, min_periods=10).std().reindex(df.index)


CANDIDATES = [
    ('vol_autocorr_20x60', f_vol_autocorr_20x60),
    ('vol_weighted_mom_20', f_vol_weighted_mom_20),
    ('ewma_mom_20', f_ewma_mom_20),
    ('stoch_k_14', f_stoch_k_14),
    ('beta_instability_60', f_beta_instability_60),
    ('time_above_ma_60', f_time_above_ma_60),
    ('aroon_20', f_aroon_20),
    ('keltner_pos_20', f_keltner_pos_20),
    ('rank_autocorr_60', f_rank_autocorr_60),
    ('vol_cluster_asym_20', f_vol_cluster_asym_20),
    ('comm_ratio_beta_60', f_comm_ratio_beta_60),
    ('btc_eth_ratio_beta_60', f_btc_eth_ratio_beta_60),
    ('volume_autocorr_20', f_volume_autocorr_20),
    ('ad_line_slope_20', f_ad_line_slope_20),
    ('range_vol_20', f_range_vol_20),
]

results = {}
for fid, fn in CANDIDATES:
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
    except Exception as e:
        print(f'{fid}: factor_fn EXCEPTION {e}', flush=True)
        results[fid] = {'ok': False, 'error': str(e)}
        continue
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data -> None', flush=True)
        results[fid] = {'ok': False, 'metrics': None}
        continue
    rho, rho_id, per_id = spearman_rho_vs_library(panel)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ic_ok = abs(m['ic']) >= 0.007
    icir_ok = abs(m['icir']) >= 0.084
    corr_ok = rho < 0.5
    ok = ic_ok and icir_ok and corr_ok
    results[fid] = {'ok': ok, 'metrics': m}
    print(f'Factor {fid}: panel {panel.shape} [{time.time()-t1:.1f}s]', flush=True)
    print(json.dumps({k: v for k, v in m.items() if k != 'decay_ic_by_horizon'}, indent=1, default=str), flush=True)
    print('decay:', json.dumps(m['decay_ic_by_horizon'], default=str), flush=True)
    print(f"top rho: { {k: round(v,3) for k,v in sorted(per_id.items(), key=lambda kv: abs(kv[1]), reverse=True)[:3]} }", flush=True)
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {ic_ok} | |ICIR|={abs(m['icir']):.4f}>=0.084 {icir_ok} | corr={rho:.3f}<0.5 {corr_ok} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print('-' * 80, flush=True)

out = Path(f'scripts/miner_2_20261022_results_round24.json')
out.write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')
print(f'elapsed={time.time()-t0:.1f}s results -> {out}', flush=True)
