"""miner_2 round-13 novel factor screen (fresh ideas, not in prior batches).

Candidates:
  - hurst_64           : Hurst exponent of log-price (variance-ratio style, scales 1/2/4/8 over 64d)
  - leadlag_spx_20     : 20d corr(r_asset_t, r_spx_{t-1}) -- delayed reaction to market
  - ew_beta_60         : 60d beta vs leave-one-out equal-weight cross-asset index (systemic beta)
  - cross_corr_20      : 20d mean pairwise return correlation with other assets (centrality)
  - sharpe_term_20_60  : 20d Sharpe - 60d Sharpe (short/medium risk-adjusted momentum term structure)
  - obv_diverg_20      : OBV 20d slope (z) minus price 20d return (z) -- volume/price divergence
  - dist_low_252       : (close - 252d min low) / (252d max high - 252d min low)
  - down_up_beta_60    : downside beta / upside beta ratio vs SPX (60d)

Admission gate: |IC10| >= 0.007, |ICIR10| >= 0.084, max |library rho| < 0.5.
"""
import sys, time
sys.path.insert(0, 'scripts')
import numpy as np
import pandas as pd
from pathlib import Path
from factor_common import (load_prices, factor_to_panel, validate_factor,
                           canonical_grid, signal_matrix, WATCHLIST)

t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
print(f'prices={len(prices)} grid={len(grid)} {grid.min().date()}..{grid.max().date()}', flush=True)

# ---- load full library signal artifacts (13 effective factors) ----
lib = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    fid = p.name.replace('_signal.npy', '')
    lib[fid] = np.load(p, allow_pickle=False)
print(f'library factors: {len(lib)}: {sorted(lib)}', flush=True)


def lib_max_corr(panel):
    arr = signal_matrix(panel, grid)
    best, best_id = 0.0, None
    for fid, larr in lib.items():
        corrs = []
        for i in range(arr.shape[0]):
            x, y = arr[i], larr[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xr = pd.Series(x[m]).rank().values
                yr = pd.Series(y[m]).rank().values
                c = np.corrcoef(xr, yr)[0, 1]
                if np.isfinite(c):
                    corrs.append(c)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id


# ---- precomputed cross-asset returns panel (for index/centrality factors) ----
ret_panel = pd.DataFrame({s: df['close'].pct_change() for s, df in prices.items()}).sort_index()
spx_ret = ret_panel['SPX']

# leave-one-out equal-weight index return
ew_ret = (ret_panel.sum(axis=1, min_count=1) - ret_panel) / ret_panel.notna().sum(axis=1).clip(lower=1).sub(1).where(ret_panel.notna().sum(axis=1) > 1, 1)

# ---------------- candidates ----------------
def f_hurst_64(df, s):
    r = df['close'].pct_change()
    out = {}
    for i in range(64, len(r)):
        w = r.iloc[i-64:i].values
        w = w[np.isfinite(w)]
        if len(w) < 64:
            continue
        stds = []
        for scale in (1, 2, 4, 8):
            n = len(w) // scale
            blocks = w[:n*scale].reshape(n, scale).sum(axis=1)
            stds.append(blocks.std(ddof=1))
        stds = np.array(stds)
        if np.all(stds > 0) and np.isfinite(stds).all():
            out[r.index[i]] = np.polyfit(np.log([1, 2, 4, 8]), np.log(stds), 1)[0]
    return pd.Series(out)

def f_leadlag_spx_20(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), spx_ret.shift(1).rename('s')], axis=1)
    return z['r'].rolling(20).corr(z['s'])

def f_ew_beta_60(df, s):
    r = df['close'].pct_change()
    e = ew_ret[s]
    z = pd.concat([r.rename('r'), e.rename('e')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['e']) / z['e'].rolling(60).var()
    return b

def f_cross_corr_20(df, s):
    r = df['close'].pct_change()
    others = [c for c in WATCHLIST if c != s and c in ret_panel.columns]
    o = ret_panel[others]
    c = r.rolling(20).corr(o).mean(axis=1, skipna=True)
    return c

def f_sharpe_term_20_60(df, s):
    r = df['close'].pct_change()
    mu20 = r.rolling(20).mean(); sd20 = r.rolling(20).std()
    mu60 = r.rolling(60).mean(); sd60 = r.rolling(60).std()
    sh20 = mu20 / sd20
    sh60 = mu60 / sd60
    return (sh20 - sh60)

def f_obv_diverg_20(df, s):
    r = df['close'].pct_change()
    obv = (np.sign(r) * df['volume'].astype(float)).fillna(0).cumsum()
    obv_slope = obv.diff(20)
    obv_z = (obv_slope - obv_slope.rolling(120).mean()) / obv_slope.rolling(120).std()
    pr_z = (r.rolling(20).sum() - r.rolling(20).sum().rolling(120).mean()) / r.rolling(20).sum().rolling(120).std()
    return obv_z - pr_z

def f_dist_low_252(df, s):
    hl = df['high'].rolling(252).max()
    ll = df['low'].rolling(252).min()
    rng = (hl - ll)
    return ((df['close'] - ll) / rng).where(rng > 0)

def f_down_up_beta_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), spx_ret.rename('s')], axis=1).dropna()
    z['down'] = (z['s'] < 0).astype(float)
    z['up'] = (z['s'] > 0).astype(float)
    bd = z['r'].rolling(60).cov(z['s'] * z['down']) / (z['s'] * z['down']).rolling(60).var()
    bu = z['r'].rolling(60).cov(z['s'] * z['up']) / (z['s'] * z['up']).rolling(60).var()
    return bd / bu.abs()

candidates = {
    'hurst_64': f_hurst_64,
    'leadlag_spx_20': f_leadlag_spx_20,
    'ew_beta_60': f_ew_beta_60,
    'cross_corr_20': f_cross_corr_20,
    'sharpe_term_20_60': f_sharpe_term_20_60,
    'obv_diverg_20': f_obv_diverg_20,
    'dist_low_252': f_dist_low_252,
    'down_up_beta_60': f_down_up_beta_60,
}

results = {}
for fid, fn in candidates.items():
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
        m = validate_factor(fid, panel, prices)
        if m is None:
            print(f'{fid}: insufficient data -> skip', flush=True)
            continue
        rho, rid = lib_max_corr(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rid
        ok_ic = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084
        ok_corr = rho < 0.5
        print(f"{fid}: ic={m['ic']:+.4f} icir={m['icir']:+.4f} hit={m['ic_hit_ratio']:.3f} "
              f"n={m['n_ic_dates']} cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rid}) "
              f"d1={m['decay_ic_by_horizon']['1']:+.4f} d5={m['decay_ic_by_horizon']['5']:+.4f} "
              f"d10={m['decay_ic_by_horizon']['10']:+.4f} d20={m['decay_ic_by_horizon']['20']:+.4f} "
              f"-> {'PASS' if (ok_ic and ok_corr) else 'skip'} [{time.time()-t1:.1f}s]", flush=True)
        results[fid] = (m, panel)
    except Exception as e:
        print(f'{fid}: ERROR {type(e).__name__}: {e}', flush=True)

print(f'\nTOTAL {time.time()-t0:.1f}s')
print('SUMMARY:')
for fid, (m, _) in sorted(results.items()):
    ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and m['max_abs_library_correlation'] < 0.5
    print(f"  {fid:20s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} rho={m['max_abs_library_correlation']:.3f} -> {'ADMIT' if ok else 'skip'}")
