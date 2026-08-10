"""miner_1 2026-07-30 batch-5 screening (FAST vectorized): NEW orthogonal factor families.

Same 10 candidates as miner_1_20260730_screen_batch5.py but with numpy-vectorized
daily cross-sectional Spearman (rank matrices + Pearson-on-ranks) so it completes
in well under the 300s shell budget.

Library (12 EFFECTIVE): copper_beta_60, dd_duration_120_resid, dxy_beta_cond_60x20,
eurusd_beta_cond_60x20, hilo_pos_60, hs300_beta_60, max_ret_20d, skew_term_20_60,
spx_beta_60, vix_beta_cond_60x20, vol_adj_mom_20_60, vol_of_vol20x60.

Candidates (all NEW, interpretable):
  1. cvar_20        : 20d 5% CVaR of daily returns
  2. coskew_60      : 60d co-skewness of asset ret with SPX ret
  3. down_beta_60   : SPX beta on market-down days only
  4. beta_asym_60   : downside-beta minus upside-beta
  5. parkinson_vol_20 : 20d high-low range vol estimator
  6. sharpe_60      : 60d mean daily ret / std
  7. accel_mom_60_20: 60d mom minus 20d mom (acceleration)
  8. min_ret_20d    : worst daily return over 20d
  9. kurtosis_20    : 20d realized excess kurtosis
 10. treynor_60     : 60d ret / 60d SPX beta

Gate: |IC(h=10)| >= 0.007 AND |ICIR| >= 0.084 on 2020-01-01..2026-07-15;
rho vs 12-factor artifact library < 0.5 for gate survival.
"""
import sys, json, time
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, factor_to_panel, WATCHLIST,
                           canonical_grid, VAL_START, VAL_END)

T0 = time.time()
def log(msg):
    print(f'[{time.time()-T0:6.1f}s] {msg}', flush=True)

prices = load_prices(days=2200)
spx = load_index('SPX', prices=prices) or prices.get('SPX')
grid = canonical_grid(prices)
log(f'load: {len(prices)} assets; grid n={len(grid)} {grid.min().date()}..{grid.max().date()}')

# ---------------- library panels (artifacts + 3 rebuilds) ----------------
EFFECTIVE = ['copper_beta_60', 'dd_duration_120_resid', 'dxy_beta_cond_60x20',
             'eurusd_beta_cond_60x20', 'hilo_pos_60', 'hs300_beta_60',
             'max_ret_20d', 'skew_term_20_60', 'spx_beta_60',
             'vix_beta_cond_60x20', 'vol_adj_mom_20_60', 'vol_of_vol20x60']
EURUSD = load_index('EURUSD', prices=prices)
VIX = load_index('VIX', prices=prices)

lib_panels = {}
for fid in EFFECTIVE:
    p = f'factors/{fid}_signal.npy'
    try:
        arr = np.load(p, allow_pickle=False)
        lib_panels[fid] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
        continue
    except Exception:
        pass
    if fid == 'eurusd_beta_cond_60x20':
        def f(df, s):
            if EURUSD is None: return None
            r = df['close'].pct_change(); vr = EURUSD['close'].pct_change()
            z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
            b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
            return (b * (EURUSD['close'] / EURUSD['close'].shift(20) - 1.0)).reindex(z.index)
    elif fid == 'hilo_pos_60':
        def f(df, s):
            return (df['close'] - df['low'].rolling(60).min()) / (df['high'].rolling(60).max() - df['low'].rolling(60).min())
    elif fid == 'vix_beta_cond_60x20':
        def f(df, s):
            if VIX is None: return None
            r = df['close'].pct_change(); vr = VIX['close'].pct_change()
            z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
            b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
            return (-b * (VIX['close'] / VIX['close'].shift(20) - 1.0)).reindex(z.index)
    else:
        continue
    lib_panels[fid] = factor_to_panel(f, prices)
log(f'library panels: {len(lib_panels)} ({list(lib_panels.keys())})')

# ---------------- vectorized daily Spearman helpers ----------------
def rank_mat(panel):
    """(n_dates, 15) ranks per date (NaN preserved), aligned to grid."""
    p = panel.reindex(grid)
    r = p.rank(axis=1)          # NaN -> NaN
    return r[WATCHLIST].values.astype(float)

def daily_spearman(a, b, min_valid=8):
    """Daily Pearson-on-ranks between two (n,15) rank matrices (NaN aware)."""
    n = a.shape[0]
    out = np.full(n, np.nan)
    mask = np.isfinite(a) & np.isfinite(b)
    cnt = mask.sum(axis=1)
    ok = cnt >= min_valid
    idx = np.where(ok)[0]
    for i in idx:
        x = a[i, mask[i]]; y = b[i, mask[i]]
        x = x - x.mean(); y = y - y.mean()
        den = np.sqrt((x*x).sum() * (y*y).sum())
        out[i] = (x*y).sum() / den if den > 0 else np.nan
    return out

def ic_series(rank, fwd_rank, min_valid=8):
    return daily_spearman(rank, fwd_rank, min_valid)

def validate_fast(fid, panel, prices):
    """Mirror of factor_common.validate_factor but vectorized."""
    horizons = (1, 2, 3, 5, 10, 20)
    rk = rank_mat(panel)
    fwd_ranks = {}
    for h in horizons:
        fr = pd.DataFrame({s: df['close'].shift(-h) / df['close'] - 1.0
                           for s, df in prices.items()}).sort_index()
        fwd_ranks[h] = rank_mat(fr)
    ic_s = {h: ic_series(rk, fwd_ranks[h]) for h in horizons}

    ic10 = ic_s[10]
    # restrict to validation window (grid already in window; drop NaN)
    ic10 = ic10[np.isfinite(ic10)]
    if len(ic10) < 100:
        return None
    ic_mean = float(ic10.mean())
    ic_std = float(ic10.std(ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())

    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    total_cells = fac.shape[0] * fac.shape[1]
    valid_cells = int(fac.notna().sum().sum())
    coverage = valid_cells / total_cells if total_cells else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())

    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')

    decay = {str(h): float(np.nanmean(ic_s[h])) for h in horizons}
    return {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit,
            'n_ic_dates': int(len(ic10)), 'coverage_asset_days': coverage,
            'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
            'decay_ic_by_horizon': decay}

def max_lib_corr(rank, lib_ranks, min_valid=8):
    best, best_id = 0.0, None
    for fid, lr in lib_ranks.items():
        ic = daily_spearman(rank, lr, min_valid)
        r = float(np.nanmean(ic)) if np.isfinite(ic).any() else 0.0
        if abs(r) > best:
            best, best_id = abs(r), fid
    return best, best_id

log('precomputing library rank matrices...')
lib_ranks = {fid: rank_mat(pan) for fid, pan in lib_panels.items()}

# ---------------- candidate definitions ----------------
def f_cvar20(df, s):
    return df['close'].pct_change().rolling(20).quantile(0.05)

def f_coskew60(df, s):
    r = df['close'].pct_change(); rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    mu_r = z['r'].rolling(60).mean(); mu_m = z['m'].rolling(60).mean()
    sr = z['r'].rolling(60).std(); sm = z['m'].rolling(60).std()
    num = ((z['r'] - mu_r) * (z['m'] - mu_m) ** 2).rolling(60).mean()
    return (num / (sr * sm ** 2)).reindex(z.index)

def f_down_beta60(df, s):
    r = df['close'].pct_change(); rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    down = z[z['m'] < 0]
    b = down['r'].rolling(60).cov(down['m']) / down['m'].rolling(60).var()
    return b.reindex(z.index)

def f_beta_asym60(df, s):
    r = df['close'].pct_change(); rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    down = z[z['m'] < 0]; up = z[z['m'] >= 0]
    bd = down['r'].rolling(60).cov(down['m']) / down['m'].rolling(60).var()
    bu = up['r'].rolling(60).cov(up['m']) / up['m'].rolling(60).var()
    return (bd - bu).reindex(z.index)

def f_parkinson20(df, s):
    h, l = df['high'], df['low']
    rng = (np.log(h / l) ** 2 / (4 * np.log(2))).rolling(20).mean()
    return np.sqrt(rng)

def f_sharpe60(df, s):
    r = df['close'].pct_change()
    return (r.rolling(60).mean() / r.rolling(60).std()).reindex(r.index)

def f_accel_mom(df, s):
    m60 = df['close'].shift(5) / df['close'].shift(65) - 1.0
    m20 = df['close'].shift(5) / df['close'].shift(25) - 1.0
    return m60 - m20

def f_min_ret20(df, s):
    return df['close'].pct_change().rolling(20).min()

def f_kurt20(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(20).mean(); sd = r.rolling(20).std()
    return (((r - mu) ** 4).rolling(20).mean() / sd ** 4 - 3.0).reindex(r.index)

def f_treynor60(df, s):
    r = df['close'].pct_change(); rm = spx['close'].pct_change()
    z = pd.concat([r.rename('r'), rm.rename('m')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['m']) / z['m'].rolling(60).var()
    ret60 = z['r'].rolling(60).sum()
    return (ret60 / b).reindex(z.index)

CANDIDATES = [
    ('cvar_20', f_cvar20, '20d 5% CVaR tail risk'),
    ('coskew_60', f_coskew60, '60d co-skewness with SPX'),
    ('down_beta_60', f_down_beta60, 'SPX downside-only beta 60d'),
    ('beta_asym_60', f_beta_asym60, 'down minus up beta 60d'),
    ('parkinson_vol_20', f_parkinson20, '20d Parkinson (high-low) vol'),
    ('sharpe_60', f_sharpe60, '60d mean/std risk-adjusted return'),
    ('accel_mom_60_20', f_accel_mom, '60d-20d momentum acceleration'),
    ('min_ret_20d', f_min_ret20, '20d worst daily return'),
    ('kurtosis_20', f_kurt20, '20d realized excess kurtosis'),
    ('treynor_60', f_treynor60, '60d return / 60d SPX beta'),
]

results = {}
for fid, fn, desc in CANDIDATES:
    panel = factor_to_panel(fn, prices)
    m = validate_fast(fid, panel, prices)
    if m is None:
        log(f'{fid}: insufficient -> None')
        continue
    rk = rank_mat(panel)
    rho, rho_id = max_lib_corr(rk, lib_ranks)
    m['max_abs_library_correlation'] = rho
    m['max_corr_library_id'] = rho_id
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
    results[fid] = {'ok': ok, 'metrics': m}
    print(f"\n=== {fid} ({desc}) shape={panel.shape} ===", flush=True)
    print(f"  IC={m['ic']:.4f} ICIR={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
          f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} "
          f"cov8={m['coverage_dates_ge8']:.3f} turn={m['turnover_10d_rank']:.3f} "
          f"maxlibrho={rho:.3f}({rho_id})", flush=True)
    print(f"  decay={ {h: round(v,4) for h, v in m['decay_ic_by_horizon'].items()} }", flush=True)
    print(f"  ADMISSION: {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f} "
          f"{'ok' if abs(m['ic'])>=0.007 else 'NO'}, |ICIR|={abs(m['icir']):.4f} "
          f"{'ok' if abs(m['icir'])>=0.084 else 'NO'})", flush=True)

with open('scripts/miner_1_20260730_results_batch5.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)
log(f'[done] saved scripts/miner_1_20260730_results_batch5.json in {time.time()-T0:.1f}s')
