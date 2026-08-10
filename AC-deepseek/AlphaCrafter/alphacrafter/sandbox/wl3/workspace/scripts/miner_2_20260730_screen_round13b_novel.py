"""Debug leadlag_spx_20 / ew_beta_60 + second batch of fresh candidates (round 13b)."""
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

lib = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    fid = p.name.replace('_signal.npy', '')
    lib[fid] = np.load(p, allow_pickle=False)
print(f'library factors: {len(lib)}', flush=True)

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

ret_panel = pd.DataFrame({s: df['close'].pct_change() for s, df in prices.items()}).sort_index()
spx_ret = ret_panel['SPX']
btc_ret = ret_panel['BTC']
vix_df = None
try:
    from factor_common import load_index
    vix_df = load_index('VIX', prices=prices)
except Exception as e:
    print('vix load err', e)

# clean leave-one-out EW index
sum_all = ret_panel.sum(axis=1, min_count=1)
cnt = ret_panel.notna().sum(axis=1)
ew_cols = {}
for s in WATCHLIST:
    ew_cols[s] = ((sum_all - ret_panel[s]) / (cnt - 1)).where(cnt > 1)
ew_ret = pd.DataFrame(ew_cols).sort_index()

# ---------------- batch 2 candidates ----------------
def f_leadlag_btc_20(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), btc_ret.shift(1).rename('b')], axis=1)
    return z['r'].rolling(20).corr(z['b'])

def f_trend_strength_20(df, s):
    c = np.log(df['close'])
    x = np.arange(20)
    def r2_at(i):
        y = c.iloc[i-19:i+1].values
        if not np.all(np.isfinite(y)):
            return np.nan
        b = np.polyfit(x, y, 1)
        pred = np.polyval(b, x)
        ss = 1.0 - np.sum((y - pred)**2) / np.sum((y - y.mean())**2)
        return np.sign(b[0]) * ss
    idx = c.index[19:]
    return pd.Series([r2_at(i) for i in range(19, len(c))], index=idx)

def f_vix_beta_60(df, s):
    if vix_df is None:
        return None
    r = df['close'].pct_change()
    vr = vix_df['close'].pct_change()
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
    return b

def f_us10y_beta_60(df, s):
    r = df['close'].pct_change()
    u = ret_panel['US10Y']
    z = pd.concat([r.rename('r'), u.rename('u')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['u']) / z['u'].rolling(60).var()
    return b

def f_drawup_dur_120(df, s):
    roll_max = df['close'].rolling(120, min_periods=20).max()
    dd = df['close'] / roll_max - 1.0
    # duration of current drawup = distance since last drawdown trough below -1% within 120d
    trough = (dd <= -0.01)
    out = {}
    last_trough = 0
    for i, (dt, v) in enumerate(dd.items()):
        if trough.iloc[i]:
            last_trough = i
        out[dt] = i - last_trough if i > last_trough else 0
    return pd.Series(out, index=dd.index)

def f_obv_diverg_60(df, s):
    r = df['close'].pct_change()
    obv = (np.sign(r) * df['volume'].astype(float)).fillna(0).cumsum()
    obv_slope = obv.diff(60)
    obv_z = (obv_slope - obv_slope.rolling(180).mean()) / obv_slope.rolling(180).std()
    pr_z = (r.rolling(60).sum() - r.rolling(60).sum().rolling(180).mean()) / r.rolling(60).sum().rolling(180).std()
    return obv_z - pr_z

def f_ndx_beta_60(df, s):
    r = df['close'].pct_change()
    n = ret_panel['NDX']
    z = pd.concat([r.rename('r'), n.rename('n')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['n']) / z['n'].rolling(60).var()
    return b

candidates = {
    'leadlag_btc_20': f_leadlag_btc_20,
    'trend_strength_20': f_trend_strength_20,
    'vix_beta_60': f_vix_beta_60,
    'us10y_beta_60': f_us10y_beta_60,
    'drawup_dur_120': f_drawup_dur_120,
    'obv_diverg_60': f_obv_diverg_60,
    'ndx_beta_60': f_ndx_beta_60,
    'ew_beta_60': lambda df, s: (lambda r, e: pd.concat([r.rename('r'), e.rename('e')], axis=1).dropna().pipe(
        lambda z: z['r'].rolling(60).cov(z['e']) / z['e'].rolling(60).var()))(df['close'].pct_change(), ew_ret[s]),
    'leadlag_spx_20': lambda df, s: pd.concat([df['close'].pct_change().rename('r'), spx_ret.shift(1).rename('s')], axis=1)['r'].rolling(20).corr(
        pd.concat([df['close'].pct_change().rename('r'), spx_ret.shift(1).rename('s')], axis=1)['s']),
}

results = {}
for fid, fn in candidates.items():
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
        if panel.shape[1] < 5:
            print(f'{fid}: panel too thin {panel.shape} -> skip', flush=True)
            continue
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
