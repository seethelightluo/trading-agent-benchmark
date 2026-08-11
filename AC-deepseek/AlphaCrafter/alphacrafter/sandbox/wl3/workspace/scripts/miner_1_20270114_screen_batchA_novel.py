"""Batch A (2027-01-14): novel cross-asset factor screen.

Validation on warm-up 2020-01-01..2026-07-15 (canonical grid, matches library).
Gate: |IC10| >= 0.007, |ICIR10| >= 0.084, max_abs_library_correlation < 0.5
vs real signal artifacts of EFFECTIVE library factors.
Supplementary: recent online-window IC (2026-07-16..2027-01-13) reported separately.
"""
import sys, json, glob, time
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
from factor_common import (WATCHLIST, load_prices, load_index, canonical_grid,
                           signal_matrix, VAL_START, VAL_END, forward_returns)

np.seterr(all='ignore')
t0 = time.time()
prices = load_prices(days=2000)
grid = canonical_grid(prices)
T, N = len(grid), len(WATCHLIST)
val_mask = (grid >= VAL_START) & (grid <= VAL_END)
print(f"grid {T} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)} | {time.time()-t0:.1f}s", flush=True)
print(f"full data end: {max(d.index.max() for d in prices.values()).date()}", flush=True)

# ---------------- library rank matrices from real signal artifacts ----------------
lib_raw = {}
for f in sorted(glob.glob('factors/*.json')):
    try:
        d = json.load(open(f))
        if d.get('validation', {}).get('status') != 'EFFECTIVE':
            continue
        fid = d['factor_id']
        art = d.get('signal_artifact')
        if not art:
            print(f"  no artifact: {fid}", flush=True)
            continue
        arr = np.load('factors/' + art, allow_pickle=False)
        if arr.shape == (T, N):
            lib_raw[fid] = arr
    except Exception as e:
        print(f"  artifact ERR {f}: {e}", flush=True)
print(f"library artifacts: {len(lib_raw)} -> {sorted(lib_raw.keys())}", flush=True)

def rank_matrix(df):
    return df.rank(axis=1).values.astype(float)

def spearman_from_ranks(xr, yr):
    valid = np.isfinite(xr) & np.isfinite(yr)
    nv = valid.sum(axis=1)
    ok = nv >= 8
    out = np.full(len(nv), np.nan)
    xc = np.where(valid, xr, np.nan); yc = np.where(valid, yr, np.nan)
    mx = np.nanmean(xc, axis=1, keepdims=True); my = np.nanmean(yc, axis=1, keepdims=True)
    xc = np.where(valid, xr - mx, 0.0); yc = np.where(valid, yr - my, 0.0)
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    out[ok] = num[ok] / den[ok]
    return out

# sanity check of the fast spearman implementation
rng = np.random.default_rng(42)
xa = rng.normal(size=(50, 15)); xa[rng.random((50, 15)) < 0.2] = np.nan
ya = rng.normal(size=(50, 15)); ya[rng.random((50, 15)) < 0.2] = np.nan
xdf = pd.DataFrame(xa); ydf = pd.DataFrame(ya)
fast = spearman_from_ranks(rank_matrix(xdf), rank_matrix(ydf))
ref = np.array([xdf.iloc[i].corr(ydf.iloc[i], method='spearman') for i in range(50)])
print(f"SPEARMAN SANITY: max|fast-pandas|={np.nanmax(np.abs(fast-ref)):.2e} -> {'OK' if np.nanmax(np.abs(fast-ref))<1e-10 else 'MISMATCH'}", flush=True)

fwd_raw = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}
fwd_rank = {h: rank_matrix(fwd_raw[h].reindex(grid)) for h in fwd_raw}
lib_rank = {fid: rank_matrix(pd.DataFrame(arr, index=grid, columns=WATCHLIST)) for fid, arr in lib_raw.items()}

# online supplementary grid (2026-07-16 .. data end)
all_dates = sorted(set().union(*[set(d.index) for d in prices.values()]))
grid2 = pd.DatetimeIndex([d for d in all_dates if d > VAL_END])
fwd2_rank = rank_matrix(forward_returns(prices, 10).reindex(grid2))
print(f"online grid: {len(grid2)} dates {grid2.min().date()}..{grid2.max().date()}", flush=True)

# ---------------- observation index (USDJPY) ----------------
usdjpy = load_index('USDJPY', prices=prices)
print(f"USDJPY loaded: {usdjpy is not None} rows={0 if usdjpy is None else len(usdjpy)}", flush=True)

# ---------------- candidate factors ----------------
def f_usdjpy_beta_cond(df, s):
    if usdjpy is None:
        return None
    r = df['close'].pct_change()
    x = usdjpy['close'].pct_change()
    cond = usdjpy['close'] / usdjpy['close'].shift(20) - 1.0
    z = pd.concat([r.rename('r'), x.rename('x'), cond.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['x']) / z['x'].rolling(60).var().replace(0, np.nan)
    return (b * z['c']).reindex(z.index)

def f_rolling_sharpe_60(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(60).mean(); sd = r.rolling(60).std()
    return mu / sd.replace(0, np.nan)

def f_updown_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    us = (r.clip(lower=0) ** 2).rolling(20).mean().apply(np.sqrt)
    ds = (r.clip(upper=0) ** 2).rolling(20).mean().apply(np.sqrt)
    return ds / us.replace(0, np.nan)

def f_corr_change_20_60(df, s):
    r = df['close'].pct_change()
    rets = pd.DataFrame({ss: prices[ss]['close'].pct_change() for ss in WATCHLIST if ss in prices})
    basket = rets.mean(axis=1)
    z = pd.concat([r.rename('r'), basket.rename('b')], axis=1).dropna()
    c20 = z['r'].rolling(20).corr(z['b'])
    c60 = z['r'].rolling(60).corr(z['b'])
    return c20 - c60

def make_ratio_beta(sym_a, sym_b, win):
    pa = prices[sym_a]['close']; pb = prices[sym_b]['close']
    xr = (pa / pb).pct_change()
    def f(df, s):
        r = df['close'].pct_change()
        z = pd.concat([r.rename('r'), xr.rename('x')], axis=1).dropna()
        b = z['r'].rolling(win).cov(z['x']) / z['x'].rolling(win).var().replace(0, np.nan)
        return b
    return f

def f_boll_bandwidth_20(df, s):
    c = df['close']
    ma = c.rolling(20).mean(); sd = c.rolling(20).std()
    return (2.0 * sd) / ma.replace(0, np.nan)

def f_kurt_std_20(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).kurt()

def f_breakout_frac_10(df, s):
    c = df['close']; h = df['high']
    brk = (c > h.shift(1)).astype(float)
    return brk.rolling(10).mean()

def f_downside_beta_20(df, s):
    spx = prices['SPX']['close'].pct_change()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), spx.rename('m')], axis=1).dropna()
    dn = z[z['m'] < 0]
    if len(dn) < 30:
        return None
    b = dn['r'].rolling(20).cov(dn['m']) / dn['m'].rolling(20).var().replace(0, np.nan)
    return b

def f_zscore_20(df, s):
    c = df['close']
    mu = c.rolling(20).mean(); sd = c.rolling(20).std()
    return (c - mu) / sd.replace(0, np.nan)

def f_yield_spread_beta_60(df, s):
    cn = prices['CN10Y']['close']; us = prices['US10Y']['close']
    spread = cn - us
    xr = spread.diff()
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), xr.rename('x')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['x']) / z['x'].rolling(60).var().replace(0, np.nan)
    return b

def f_rel_vol_20(df, s):
    r = df['close'].pct_change()
    v = r.rolling(20).std()
    vols = pd.DataFrame({ss: prices[ss]['close'].pct_change().rolling(20).std() for ss in WATCHLIST if ss in prices})
    med = vols.median(axis=1)
    return v / med.replace(0, np.nan)

def f_up_capture_20(df, s):
    r = df['close'].pct_change()
    up = r[r > 0].rolling(20).mean()
    dn = r[r < 0].rolling(20).mean().abs()
    return up / dn.replace(0, np.nan)

cands = {
    'usdjpy_beta_cond_60x20': (f_usdjpy_beta_cond, 'beta of asset ret to USDJPY ret (60d) x 20d USDJPY move', 'FX carry/haven conditional beta'),
    'rolling_sharpe_60':      (f_rolling_sharpe_60, 'mean(ret,60)/std(ret,60)', 'risk-adjusted momentum'),
    'updown_vol_ratio_20':    (f_updown_vol_ratio_20, 'downside semidev / upside semidev (20d)', 'vol asymmetry'),
    'corr_change_20_60':      (f_corr_change_20_60, '20d corr to basket minus 60d corr', 'correlation regime shift'),
    'wti_copper_beta_60':     (make_ratio_beta('WTI', 'COPPER', 60), 'beta of asset ret to d(log WTI/COPPER) (60d)', 'energy vs metals ratio beta'),
    'btc_eth_ratio_beta_60':  (make_ratio_beta('BTC', 'ETH', 60), 'beta of asset ret to d(log BTC/ETH) (60d)', 'crypto rotation beta'),
    'spx_hsi_ratio_beta_60':  (make_ratio_beta('SPX', 'HSI', 60), 'beta of asset ret to d(log SPX/HSI) (60d)', 'DM vs CN equity ratio beta'),
    'boll_bandwidth_20':      (f_boll_bandwidth_20, '(2*std20)/MA20', 'normalized volatility level'),
    'kurt_std_20':            (f_kurt_std_20, 'excess kurtosis of 20d daily returns', 'tail thickness'),
    'breakout_frac_10':       (f_breakout_frac_10, 'frac of 10d closes above prior-day high', 'breakout frequency'),
    'downside_beta_20':       (f_downside_beta_20, 'beta to SPX on down-days only (20d)', 'short downside beta'),
    'zscore_20':              (f_zscore_20, '(close-MA20)/std20', 'short-term mean reversion z-score'),
    'yield_spread_beta_60':   (f_yield_spread_beta_60, 'beta of asset ret to d(CN10Y-US10Y) (60d)', 'yield spread beta'),
    'rel_vol_20':             (f_rel_vol_20, 'asset 20d vol / cross-sectional median 20d vol', 'relative volatility'),
    'up_capture_20':          (f_up_capture_20, 'mean up-day ret / mean |down-day ret| (20d)', 'gain/loss capture asymmetry'),
}

def factor_to_panel(fn, prices):
    cols = {}
    for s, df in prices.items():
        try:
            ser = fn(df, s)
            if ser is not None and len(ser) > 0:
                cols[s] = ser.astype(float)
        except Exception as e:
            print(f"    {s} ERR {e}", flush=True)
    if not cols:
        return pd.DataFrame()
    panel = pd.DataFrame(cols)
    return panel[~panel.index.duplicated(keep='last')].sort_index()

def fast_validate(fid, panel):
    pm = signal_matrix(panel, grid)
    pr = rank_matrix(pd.DataFrame(pm, index=grid, columns=WATCHLIST))
    ic10 = spearman_from_ranks(pr, fwd_rank[10])[val_mask]
    if np.isfinite(ic10).sum() < 100:
        return None
    ic_mean = float(np.nanmean(ic10))
    ic_std = float(np.nanstd(ic10, ddof=1))
    icir = ic_mean / ic_std if ic_std > 0 else 0.0
    hit = float((ic10 > 0).mean()) if ic_mean >= 0 else float((ic10 < 0).mean())
    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    total_cells = fac.shape[0] * fac.shape[1]
    coverage = float(fac.notna().sum().sum()) / total_cells if total_cells else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    ranked = fac.rank(axis=1)
    turn = float(ranked.diff(10).abs().mean().mean()) if len(ranked) > 10 else float('nan')
    decay = {}
    for h in (1, 2, 3, 5, 10, 20):
        s = spearman_from_ranks(pr, fwd_rank[h])[val_mask]
        decay[str(h)] = float(np.nanmean(s))
    m = {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit,
         'n_ic_dates': int(np.isfinite(ic10).sum()), 'coverage_asset_days': coverage,
         'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
         'decay_ic_by_horizon': decay}
    gv = grid[val_mask]
    for nm, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                     ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                     ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic10[(gv >= pd.Timestamp(a)) & (gv <= pd.Timestamp(b))]
        m[nm] = float(np.nanmean(sub)) if np.isfinite(sub).sum() > 30 else float('nan')
    # online supplementary (2026-07-16..data end)
    pr2 = rank_matrix(pd.DataFrame(panel.reindex(grid2), index=grid2, columns=WATCHLIST))
    ic2 = spearman_from_ranks(pr2, fwd2_rank)
    ic2 = ic2[np.isfinite(ic2)]
    if len(ic2) >= 20:
        m['online_ic'] = float(np.nanmean(ic2))
        sd2 = float(np.nanstd(ic2, ddof=1))
        m['online_icir'] = float(np.nanmean(ic2) / sd2) if sd2 > 0 else 0.0
        m['online_n'] = int(len(ic2))
    return m

def max_lib_corr(panel):
    pm = signal_matrix(panel, grid)
    pr = rank_matrix(pd.DataFrame(pm, index=grid, columns=WATCHLIST))
    best, best_id, per_id = 0.0, None, {}
    for fid, lr in lib_rank.items():
        r = spearman_from_ranks(pr, lr)
        rr = r[np.isfinite(r)]
        if len(rr) > 0:
            per_id[fid] = float(np.mean(rr))
            if abs(per_id[fid]) > best:
                best, best_id = abs(per_id[fid]), fid
    return best, best_id, per_id

results = {}
for fid, (fn, desc, tag) in cands.items():
    t1 = time.time()
    try:
        panel = factor_to_panel(fn, prices)
        if panel is None or len(panel) == 0:
            print(f"{fid}: EMPTY panel -> skip", flush=True)
            continue
        m = fast_validate(fid, panel)
        if m is None:
            print(f"{fid}: insufficient data -> None", flush=True)
            continue
        rho, rho_id, per_id = max_lib_corr(panel)
        m['max_abs_library_correlation'] = rho
        m['max_corr_library_id'] = rho_id
        m['per_factor_rho'] = {k: round(v, 3) for k, v in sorted(per_id.items(), key=lambda kv: -abs(kv[1]))[:4]}
        ok = abs(m['ic']) >= 0.007 and abs(m['icir']) >= 0.084 and rho < 0.5
        results[fid] = {'ok': bool(ok), 'metrics': {k: v for k, v in m.items()}, 'desc': desc, 'tag': tag}
        dec = {h: round(v, 4) for h, v in m['decay_ic_by_horizon'].items()}
        print(f"\n{fid}: IC10={m['ic']:.4f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} "
              f"cov={m['coverage_asset_days']:.3f} ge8={m['coverage_dates_ge8']:.3f} "
              f"turn={m['turnover_10d_rank']:.2f} rho={rho:.3f}({rho_id}) [{time.time()-t1:.1f}s]", flush=True)
        print(f"  decay: {dec}", flush=True)
        print(f"  top rho: {m['per_factor_rho']}", flush=True)
        for nm in ['ic_2020_2022', 'ic_2023_2024', 'ic_2025_2026']:
            print(f"  {nm}: {m.get(nm, float('nan')):.4f}", flush=True)
        if 'online_ic' in m:
            print(f"  online: ic={m['online_ic']:.4f} icir={m['online_icir']:.4f} n={m['online_n']}", flush=True)
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007 |ICIR|={abs(m['icir']):.4f}/0.084 rho={rho:.3f}/0.5)", flush=True)
    except Exception as e:
        print(f"{fid}: EXCEPTION {e}", flush=True)
        results[fid] = {'ok': False, 'error': str(e), 'desc': desc, 'tag': tag}

with open('scripts/miner_1_20270114_results_batchA.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)

print("\n=== SUMMARY ===")
for fid, r in sorted(results.items()):
    if 'metrics' in r:
        m = r['metrics']
        print(f"{fid:24s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m.get('max_abs_library_correlation', float('nan')):.3f} ({m.get('max_corr_library_id')}) online={m.get('online_ic', float('nan')):.4f}")
    else:
        print(f"{fid:24s} ERROR {r.get('error', '')[:80]}")
print(f"total time {time.time()-t0:.1f}s")
