"""Round 20b (2026-09-10): robust re-run of round-20 novel factor screen.

Fixes:
  - down_vol_ratio_20 uses proper downside/upside semi-deviation (rolling mean of
    squared deviations instead of std over boolean-filtered series).
  - every candidate is wrapped in try/except so one broken construction cannot
    kill the batch.
  - sub-period ICs (2020-22 / 2023-24 / 2025-26) and recent 1y IC/ICIR are
    computed from the IC10 series for full regime diagnostics.

Gate: |IC10| >= 0.007, |ICIR10| >= 0.084, max_abs_library_correlation < 0.5.
"""
import sys, json, glob
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, INDEX_SIGNALS, load_prices, load_index,
                           canonical_grid, factor_to_panel, validate_factor,
                           signal_matrix, VAL_START, VAL_END, forward_returns,
                           rank_ic_series)

np.seterr(all='ignore')
prices = load_prices(days=2500)
grid = canonical_grid(prices)
print(f"grid {len(grid)} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)}", flush=True)

# ---------------- full library rank matrices ----------------
def rank_matrix(arr):
    out = np.full(arr.shape, np.nan)
    for i in range(arr.shape[0]):
        row = arr[i]
        valid = np.isfinite(row)
        if valid.sum() >= 3:
            out[i, valid] = pd.Series(row[valid]).rank().values
    return out

lib_panels = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        art = d.get('signal_artifact')
        if art:
            arr = np.load('factors/' + art, allow_pickle=False)
            if arr.shape == (len(grid), len(WATCHLIST)):
                lib_panels[fid] = pd.DataFrame(arr, index=grid, columns=WATCHLIST)
    except Exception as e:
        print(f"  artifact ERR {f}: {e}", flush=True)

vix = load_index('VIX', prices=prices)
spx_close = prices['SPX']['close'] if 'SPX' in prices else None

def f_hilo(df, s):
    return (df['close'] - df['low'].rolling(60).min()) / (df['high'].rolling(60).max() - df['low'].rolling(60).min())
def f_vixb(df, s):
    if vix is None: return None
    r = df['close'].pct_change(); vr = vix['close'].pct_change()
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['v']) / z['v'].rolling(60).var()
    return (-b * (vix['close'] / vix['close'].shift(20) - 1.0)).reindex(z.index)
def f_vov(df, s):
    return df['close'].pct_change().rolling(20).std().rolling(60).std()

for fid, fn in [('hilo_pos_60', f_hilo), ('vix_beta_cond_60x20', f_vixb), ('vol_of_vol20x60', f_vov)]:
    if fid in lib_panels:
        continue
    p = factor_to_panel(fn, prices)
    if len(p):
        lib_panels[fid] = pd.DataFrame(signal_matrix(p, grid), index=grid, columns=WATCHLIST)

print(f"library panels for rho: {len(lib_panels)} -> {sorted(lib_panels.keys())}", flush=True)

def max_lib_corr(panel):
    best, best_id = 0.0, None
    pm = signal_matrix(panel, grid)
    for fid, lp in lib_panels.items():
        lm = lp.values
        corrs = []
        for t in range(len(grid)):
            x = pm[t]; y = lm[t]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                xv = x[m]; yv = y[m]
                xc = xv - xv.mean(); yc = yv - yv.mean()
                den = np.sqrt((xc * xc).sum() * (yc * yc).sum())
                if den > 0:
                    corrs.append((xc * yc).sum() / den)
        if corrs:
            r = float(np.mean(corrs))
            if abs(r) > best:
                best, best_id = abs(r), fid
    return best, best_id

# ---------------- candidates ----------------
us_ret = prices['SPX']['close'].pct_change()

def make_leadlag_gap_spx(w):
    def f(df, s):
        gap = df['open'] / df['close'].shift(1) - 1.0
        z = pd.concat([gap.rename('g'), us_ret.shift(1).rename('u')], axis=1).dropna()
        return z['g'].rolling(w).corr(z['u'])
    return f

def make_cskew_spx(w):
    def f(df, s):
        r = df['close'].pct_change()
        m = us_ret
        z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
        ri = z['r']; mi = z['m']
        num = ((ri - ri.mean()) * (mi - mi.mean()) ** 2).rolling(w).mean()
        den = ri.rolling(w).std() * (mi.rolling(w).std() ** 2)
        return (num / den).reindex(z.index)
    return f

def make_yldspread_beta(w):
    def f(df, s):
        u10 = prices['US10Y']['close']
        c10 = prices['CN10Y']['close']
        spread = (u10 - c10).dropna()
        ds = spread.diff()
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), ds.rename('d')], axis=1).dropna()
        b = z['r'].rolling(w).cov(z['d']) / z['d'].rolling(w).var()
        return b.reindex(z.index)
    return f

def make_parkinson_ratio(w):
    def f(df, s):
        cc = df['close'].pct_change()
        cc_vol = cc.rolling(w).std()
        hl = np.log(df['high'] / df['low'])
        pk = (hl ** 2).rolling(w).mean() ** 0.5 / np.sqrt(4 * np.log(2))
        return (pk / cc_vol.replace(0, np.nan))
    return f

def make_vwap_dev(w):
    def f(df, s):
        tp = (df['high'] + df['low'] + df['close']) / 3.0
        v = df['volume'].replace(0, np.nan)
        vwap = (tp * v).rolling(w).sum() / v.rolling(w).sum()
        return (df['close'] / vwap - 1.0)
    return f

def make_bb_width(w):
    def f(df, s):
        c = df['close']
        mid = c.rolling(w).mean()
        sd = c.rolling(w).std()
        return (4 * sd) / mid.replace(0, np.nan)
    return f

def make_gap_persist(w):
    def f(df, s):
        gap = df['open'] / df['close'].shift(1) - 1.0
        return gap.rolling(w).corr(gap.shift(1))
    return f

def make_overnight_share(w):
    def f(df, s):
        gap = (df['open'] / df['close'].shift(1) - 1.0).abs()
        intra = (df['close'] / df['open'] - 1.0).abs()
        tot = gap + intra
        return (gap / tot.replace(0, np.nan)).rolling(w).mean()
    return f

def make_max_ret_asym(w):
    def f(df, s):
        r = df['close'].pct_change()
        mx = r.rolling(w).max()
        mn = r.rolling(w).min()
        sd = r.rolling(w).std()
        return ((mx - (-mn)) / sd.replace(0, np.nan))
    return f

def make_corr_drift_spx(ws, wl):
    def f(df, s):
        r = df['close'].pct_change()
        m = us_ret
        z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
        cs = z['r'].rolling(ws).corr(z['m'])
        cl = z['r'].rolling(wl).corr(z['m'])
        return (cs - cl).reindex(z.index)
    return f

def make_down_vol_ratio(w):
    """semi-deviation ratio: sqrt(mean(min(r-mean,0)^2)) / sqrt(mean(max(r-mean,0)^2))"""
    def f(df, s):
        r = df['close'].pct_change()
        mu = r.rolling(w).mean()
        dev = r - mu
        neg = dev.clip(upper=0.0) ** 2
        pos = dev.clip(lower=0.0) ** 2
        dn = neg.rolling(w).mean() ** 0.5
        up = pos.rolling(w).mean() ** 0.5
        return (dn / up.replace(0, np.nan))
    return f

def make_vol_slope(ws, wl):
    def f(df, s):
        r = df['close'].pct_change()
        s1 = r.rolling(ws).std()
        s2 = r.rolling(wl).std()
        return np.log(s1 / s2.replace(0, np.nan))
    return f

cands = {
    'leadlag_gap_spx_60': (make_leadlag_gap_spx(60), 'corr(overnight_gap, prior-day SPX ret) over 60d', 'cross-market information diffusion'),
    'cskew_spx_60': (make_cskew_spx(60), 'co-skewness of asset daily ret with SPX daily ret (60d)', 'higher-moment systematic risk'),
    'yldspread_beta_60': (make_yldspread_beta(60), 'rolling beta of asset ret on daily change of US10Y-CN10Y spread (60d)', 'yield-curve spread sensitivity'),
    'parkinson_ratio_20': (make_parkinson_ratio(20), 'Parkinson high-low vol / close-to-close vol (20d)', 'intraday vs close volatility structure'),
    'vwap_dev_10': (make_vwap_dev(10), 'close / 10d VWAP - 1 (volume-weighted typical price)', 'execution-price deviation'),
    'bb_width_20': (make_bb_width(20), 'Bollinger bandwidth (4*std20/sma20)', 'volatility squeeze/expansion'),
    'gap_persist_20': (make_gap_persist(20), 'lag-1 autocorrelation of overnight gaps (20d)', 'gap momentum/persistence'),
    'overnight_share_20': (make_overnight_share(20), 'mean |overnight gap| / (|gap|+|intraday|) over 20d', 'overnight return share'),
    'max_ret_asym_20': (make_max_ret_asym(20), '(max daily ret + min daily ret)/vol over 20d', 'tail asymmetry'),
    'corr_drift_spx_60': (make_corr_drift_spx(60, 250), 'corr(asset,SPX,60) - corr(asset,SPX,250)', 'market-decoupling drift'),
    'down_vol_ratio_20': (make_down_vol_ratio(20), 'downside semi-dev / upside semi-dev (20d)', 'volatility asymmetry'),
    'vol_slope_20_60': (make_vol_slope(20, 60), 'log(std20/std60)', 'volatility term structure'),
}

fwd_ret = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}

def full_validate(fid, panel):
    m = validate_factor(fid, panel, prices)
    if m is None:
        return None
    ic10 = rank_ic_series(panel, fwd_ret[10], 8)
    ic10 = ic10[(ic10.index >= VAL_START) & (ic10.index <= VAL_END)]
    for nm, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                     ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                     ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic10[(ic10.index >= pd.Timestamp(a)) & (ic10.index <= pd.Timestamp(b))]
        m[nm] = float(sub.mean()) if len(sub) > 30 else float('nan')
    recent = ic10[(ic10.index >= pd.Timestamp('2025-07-15')) & (ic10.index <= pd.Timestamp('2026-07-15'))]
    if len(recent) > 30:
        m['recent_1y_ic'] = float(recent.mean())
        m['recent_1y_icir'] = float(recent.mean() / recent.std(ddof=1)) if recent.std(ddof=1) > 0 else 0.0
    return m

results = {}
for fid, (fn, desc, tag) in cands.items():
    try:
        panel = factor_to_panel(fn, prices)
        if panel is None or len(panel) == 0:
            print(f"{fid}: EMPTY panel -> skip", flush=True)
            continue
        m = full_validate(fid, panel)
        if m is None:
            print(f"{fid}: insufficient data -> None", flush=True)
            continue
        rho, rho_id = max_lib_corr(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rho_id
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
        results[fid] = {'ok': bool(ok), 'metrics': {k: v for k, v in m.items()}, 'desc': desc, 'tag': tag}
        dec = {h: round(v, 4) for h, v in m['decay_ic_by_horizon'].items()}
        print(f"\n{fid}: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
              f"coverage={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rho_id})", flush=True)
        print(f"  decay: {dec}", flush=True)
        for nm in ['ic_2020_2022', 'ic_2023_2024', 'ic_2025_2026']:
            print(f"  {nm}: {m.get(nm, float('nan')):.4f}", flush=True)
        if 'recent_1y_ic' in m:
            print(f"  recent_1y: ic={m['recent_1y_ic']:.4f} icir={m['recent_1y_icir']:.4f}", flush=True)
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007, |ICIR|={abs(m['icir']):.4f}/0.084, rho={rho:.3f}/0.5)", flush=True)
    except Exception as e:
        print(f"{fid}: EXCEPTION {e}", flush=True)
        results[fid] = {'ok': False, 'error': str(e), 'desc': desc, 'tag': tag}

with open('scripts/miner_3_20260910_results_round20b.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)

print("\n=== SUMMARY ===")
for fid, r in results.items():
    if 'metrics' in r:
        m = r['metrics']
        print(f"{fid:22s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m.get('max_abs_library_correlation', float('nan')):.3f} ({m.get('max_corr_library_id')})")
    else:
        print(f"{fid:22s} ERROR {r.get('error', '')[:80]}")
