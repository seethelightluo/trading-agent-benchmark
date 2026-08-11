"""Round 20 novel factor screen (miner_2) - 2026-09-10.

Fresh ideas (checked vs evicted/rejected/quarantine library; no duplicates):
- ret_autocorr_60        : lag-1 autocorrelation of daily returns (trend efficiency)
- candle_body_ratio_20   : mean |close-open|/(high-low) over 20d (candle efficiency)
- downside_upside_vol_60 : std(neg ret)/std(pos ret) over 60d (vol asymmetry, 2nd moment)
- mom_disp_z_20          : signed cross-sectional z-score of 20d momentum
- mom_disp_abs_20        : |cross-sectional z| of 20d momentum (extremeness)
- bias_ratio_60          : (n_up - n_down)/60 over 60d (sign balance)
- wti_beta_cond_60x20    : beta to WTI * WTI 20d momentum (conditional commodity beta)
- copper_beta_cond_60x20 : beta to COPPER * COPPER 20d momentum
- ulcer_index_60         : sqrt(mean(drawdown^2)) over 60d (drawdown depth)
- kurtosis_60            : excess kurtosis of daily returns over 60d (tail risk)
- rank_mom_smooth_5      : 5d avg of cross-sectional rank of 20d momentum
- oc_gap_vol_20          : mean |open/close_prev-1| / std(ret,20) (gap magnitude)
- dd_20_vol              : 20d drawdown from high scaled by volatility

Validates vs the full EFFECTIVE library (signal artifacts), vectorized lib-corr.
Gates: |IC|>=0.007, |ICIR|>=0.084, max_abs_library_correlation < 0.5.
"""
import sys, time, json, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, WATCHLIST, persist_factor)

TODAY = '2026-09-10'
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---- library: effective factors with signal artifacts ----
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
print(f'library factors: {len(lib)} {sorted(lib.keys())}', flush=True)


def rank_matrix(arr):
    """Row-wise Spearman ranks -> (T,15), NaN where invalid."""
    out = np.full(arr.shape, np.nan)
    T = arr.shape[0]
    for t in range(T):
        row = arr[t]
        m = np.isfinite(row)
        n = int(m.sum())
        if n >= 8:
            r = np.full(arr.shape[1], np.nan)
            r[m] = pd.Series(row[m]).rank().values
            out[t] = r
    return out


lib_rank = {fid: rank_matrix(arr) for fid, arr in lib.items()}


def lib_max_corr_fast(panel):
    arr = signal_matrix(panel, grid)
    cand_rank = rank_matrix(arr)
    best, best_id = 0.0, None
    for fid, lr in lib_rank.items():
        cs = np.full(cand_rank.shape[0], np.nan)
        for t in range(cand_rank.shape[0]):
            a, b = cand_rank[t], lr[t]
            m = np.isfinite(a) & np.isfinite(b)
            n = int(m.sum())
            if n >= 8:
                a2, b2 = a[m], b[m]
                ma, mb = a2.mean(), b2.mean()
                sa, sb = a2.std(ddof=1), b2.std(ddof=1)
                if sa > 1e-12 and sb > 1e-12:
                    cs[t] = ((a2 - ma) * (b2 - mb)).mean() / (sa * sb)
        if np.isfinite(cs).any():
            r = float(np.nanmean(cs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


# ---- benchmark/return helpers ----
def bench_ret(sym):
    return prices[sym]['close'].pct_change()


def rolling_beta(asset_df, bench_series, window, min_obs=30, bench_is_yield=False):
    r = asset_df['close'].pct_change()
    b = bench_series.diff() if bench_is_yield else bench_series
    z = pd.concat([r.rename('r'), b.rename('b')], axis=1).dropna()
    cov = z['r'].rolling(window, min_periods=min_obs).cov(z['b'])
    var = z['b'].rolling(window, min_periods=min_obs).var()
    return (cov / var).reindex(z.index)


def mom_series(sym, window=20):
    df = prices[sym]
    return df['close'] / df['close'].shift(window) - 1.0


# ---- cross-sectional momentum panels (built once) ----
def cross_mom_panels():
    panels = {}
    for s in WATCHLIST:
        if s in prices:
            panels[s] = mom_series(s, 20)
    P = pd.DataFrame(panels)
    P = P[~P.index.duplicated(keep='last')].sort_index()
    mu = P.mean(axis=1)
    sd = P.std(axis=1, ddof=1)
    rk = P.rank(axis=1)
    return P, mu, sd, rk


P_mom, XMU, XSD, XRK = cross_mom_panels()


# ---- candidate factor functions ----
def f_ret_autocorr_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=30).apply(lambda x: np.corrcoef(x[:-1], x[1:])[0, 1] if len(x) > 2 and np.std(x[:-1]) > 0 and np.std(x[1:]) > 0 else np.nan, raw=True)


def f_candle_body_ratio_20(df, s):
    rng = (df['high'] - df['low']).replace(0, np.nan)
    body = (df['close'] - df['open']).abs()
    return (body / rng).rolling(20, min_periods=10).mean()


def f_downside_upside_vol_60(df, s):
    r = df['close'].pct_change()
    neg = r.where(r < 0)
    pos = r.where(r > 0)
    sdn = neg.rolling(60, min_periods=30).std()
    sdp = pos.rolling(60, min_periods=30).std()
    return (sdn / sdp).reindex(r.index)


def f_mom_disp_z_20(df, s):
    if s not in P_mom.columns:
        return None
    z = (P_mom[s] - XMU) / XSD
    return z


def f_mom_disp_abs_20(df, s):
    if s not in P_mom.columns:
        return None
    z = (P_mom[s] - XMU) / XSD
    return z.abs()


def f_bias_ratio_60(df, s):
    r = df['close'].pct_change().dropna()
    up = r.gt(0).rolling(60, min_periods=30).sum()
    dn = r.lt(0).rolling(60, min_periods=30).sum()
    return ((up - dn) / (up + dn)).reindex(r.index)


def f_wti_beta_cond_60x20(df, s):
    if 'WTI' not in prices:
        return None
    b = rolling_beta(df, bench_ret('WTI'), 60)
    return (b * mom_series('WTI', 20)).reindex(b.index)


def f_copper_beta_cond_60x20(df, s):
    if 'COPPER' not in prices:
        return None
    b = rolling_beta(df, bench_ret('COPPER'), 60)
    return (b * mom_series('COPPER', 20)).reindex(b.index)


def f_ulcer_index_60(df, s):
    dd = df['close'] / df['close'].rolling(60, min_periods=30).max() - 1.0
    return (dd ** 2).rolling(60, min_periods=30).mean().apply(np.sqrt)


def f_kurtosis_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=30).kurt()


def f_rank_mom_smooth_5(df, s):
    if s not in XRK.columns:
        return None
    return XRK[s].rolling(5, min_periods=3).mean()


def f_oc_gap_vol_20(df, s):
    gap = (df['open'] / df['close'].shift(1) - 1.0).abs()
    r = df['close'].pct_change()
    return (gap.rolling(20, min_periods=10).mean() / r.rolling(20, min_periods=10).std()).reindex(gap.index)


def f_dd_20_vol(df, s):
    dd = df['close'] / df['close'].rolling(20, min_periods=10).max() - 1.0
    r = df['close'].pct_change()
    return (dd / r.rolling(20, min_periods=10).std()).reindex(dd.index)


CANDIDATES = [
    ('ret_autocorr_60', f_ret_autocorr_60),
    ('candle_body_ratio_20', f_candle_body_ratio_20),
    ('downside_upside_vol_60', f_downside_upside_vol_60),
    ('mom_disp_z_20', f_mom_disp_z_20),
    ('mom_disp_abs_20', f_mom_disp_abs_20),
    ('bias_ratio_60', f_bias_ratio_60),
    ('wti_beta_cond_60x20', f_wti_beta_cond_60x20),
    ('copper_beta_cond_60x20', f_copper_beta_cond_60x20),
    ('ulcer_index_60', f_ulcer_index_60),
    ('kurtosis_60', f_kurtosis_60),
    ('rank_mom_smooth_5', f_rank_mom_smooth_5),
    ('oc_gap_vol_20', f_oc_gap_vol_20),
    ('dd_20_vol', f_dd_20_vol),
]

results = {}
for fid, fn in CANDIDATES:
    t1 = time.time()
    panel = factor_to_panel(fn, prices)
    m = validate_factor(fid, panel, prices)
    if m is None:
        print(f'{fid}: insufficient data -> None', flush=True)
        results[fid] = {'ok': False, 'metrics': None}
        continue
    rho, rho_id = lib_max_corr_fast(panel)
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
    print(f"ADMISSION: |IC|={abs(m['ic']):.4f}>=0.007 {ic_ok} | |ICIR|={abs(m['icir']):.4f}>=0.084 {icir_ok} | corr={rho:.3f}<0.5 {corr_ok} -> {'PASS' if ok else 'FAIL'}", flush=True)
    print('-' * 80, flush=True)

out = Path('scripts/miner_2_20260910_results_round20.json')
out.write_text(json.dumps(results, indent=1, default=str), encoding='utf-8')
print(f'elapsed={time.time()-t0:.1f}s results -> {out}', flush=True)
