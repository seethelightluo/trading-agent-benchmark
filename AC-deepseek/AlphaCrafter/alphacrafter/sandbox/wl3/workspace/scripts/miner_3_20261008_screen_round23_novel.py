"""Round 23 (2026-10-08) batch screen for novel factors - CORRECTED validation.

Uses pandas rank(axis=1) (trusted) + numpy Pearson on ranks (fast, verified
against pandas method='spearman'). IC = daily cross-sectional Spearman of
factor vs 10d forward return over warm-up 2020-01-01..2026-07-15.
Gate: |IC10| >= 0.007, |ICIR10| >= 0.084, max_abs_library_correlation < 0.5
vs the 18 EFFECTIVE library artifacts.
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
            continue
        arr = np.load('factors/' + art, allow_pickle=False)
        if arr.shape == (T, N):
            lib_raw[fid] = arr
    except Exception as e:
        print(f"  artifact ERR {f}: {e}", flush=True)
print(f"library artifacts: {len(lib_raw)} -> {sorted(lib_raw.keys())}", flush=True)

# ---------------- numpy Spearman on precomputed rank matrices ----------------
def rank_matrix(df):
    """(T,N) float ranks via pandas rank(axis=1); NaN preserved."""
    return df.rank(axis=1).values.astype(float)

def spearman_from_ranks(xr, yr):
    """Per-date Pearson between two (T,N) rank matrices (NaN-aware, >=8 valid)."""
    valid = np.isfinite(xr) & np.isfinite(yr)
    nv = valid.sum(axis=1)
    ok = nv >= 8
    out = np.full(T, np.nan)
    xc = np.where(valid, xr, 0.0); yc = np.where(valid, yr, 0.0)
    xc -= xc.sum(axis=1, keepdims=True) / np.maximum(nv, 1)[:, None]
    yc -= yc.sum(axis=1, keepdims=True) / np.maximum(nv, 1)[:, None]
    num = (xc * yc).sum(axis=1)
    den = np.sqrt((xc * xc).sum(axis=1) * (yc * yc).sum(axis=1))
    out[ok] = num[ok] / den[ok]
    return out

# sanity check vs pandas spearman on a random sample of dates
rng = np.random.default_rng(42)
xa = rng.normal(size=(50, 15)); xa[rng.random((50, 15)) < 0.2] = np.nan
ya = rng.normal(size=(50, 15)); ya[rng.random((50, 15)) < 0.2] = np.nan
xdf = pd.DataFrame(xa); ydf = pd.DataFrame(ya)
xr = rank_matrix(xdf); yr = rank_matrix(ydf)
fast = spearman_from_ranks(xr, yr)
ref = np.array([xdf.iloc[i].corr(ydf.iloc[i], method='spearman') for i in range(50)])
diff = np.nanmax(np.abs(fast - ref))
print(f"SPEARMAN SANITY CHECK: max|fast - pandas| = {diff:.2e} -> {'OK' if diff < 1e-10 else 'MISMATCH'}", flush=True)

# ---------------- precompute forward-return rank matrices ----------------
fwd_raw = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}
fwd_rank = {h: rank_matrix(fwd_raw[h].reindex(grid)) for h in fwd_raw}

# ---------------- library rank matrices ----------------
lib_rank = {fid: rank_matrix(pd.DataFrame(arr, index=grid, columns=WATCHLIST)) for fid, arr in lib_raw.items()}

# ---------------- macro inputs ----------------
spx = prices['SPX']['close']
spx_ret = spx.pct_change()
ndx = prices['NDX']['close']
us10y = prices['US10Y']['close']
cn10y = prices['CN10Y']['close']
jpy = load_index('USDJPY', prices=prices)

def f_trend_r2_60(df, s):
    c = np.log(df['close'])
    idx = pd.Series(np.arange(len(c)), index=c.index)
    return c.rolling(60).corr(idx) ** 2

def f_sortino_60(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(60).mean()
    neg = r.where(r < 0)
    dd = np.sqrt((neg ** 2).rolling(60).mean())
    return mu / dd.replace(0, np.nan)

def f_profit_factor_60(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0, 0.0).rolling(60).sum()
    dn = r.where(r < 0, 0.0).rolling(60).sum()
    return up / dn.abs().replace(0, np.nan)

def f_us10y_beta_60(df, s):
    r = df['close'].pct_change()
    dy = us10y.diff()
    z = pd.concat([r.rename('r'), dy.rename('y')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['y']) / z['y'].rolling(60).var().replace(0, np.nan)
    return b.reindex(z.index)

def f_us10y_corr_60(df, s):
    r = df['close'].pct_change()
    dy = us10y.diff()
    z = pd.concat([r.rename('r'), dy.rename('y')], axis=1).dropna()
    return z['r'].rolling(60).corr(z['y']).reindex(z.index)

def f_rates_spread_beta_60(df, s):
    r = df['close'].pct_change()
    ds = (us10y - cn10y).diff()
    z = pd.concat([r.rename('r'), ds.rename('s')], axis=1).dropna()
    b = z['r'].rolling(60).cov(z['s']) / z['s'].rolling(60).var().replace(0, np.nan)
    return b.reindex(z.index)

def f_idio_mom_60_20(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), spx_ret.rename('m')], axis=1).dropna()
    # rolling 60d beta then residual = r - beta*m (using trailing window)
    b = z['r'].rolling(60).cov(z['m']) / z['m'].rolling(60).var().replace(0, np.nan)
    resid = (z['r'] - b * z['m']).dropna()
    return resid.rolling(60).sum().reindex(z.index)

def f_tail_ratio_60(df, s):
    r = df['close'].pct_change()
    up = r.rolling(60).quantile(0.95)
    dn = r.rolling(60).quantile(0.05)
    return up / dn.abs().replace(0, np.nan)

def f_range_vol_adj_mom_20(df, s):
    c = df['close']
    mom = c / c.shift(20) - 1.0
    hl = np.log(df['high'] / df['low'])
    park = np.sqrt((hl ** 2).rolling(20).mean() / (4 * np.log(2)))
    return mom / park.replace(0, np.nan)

def f_close_ma60_pos(df, s):
    c = df['close']
    return c / c.rolling(60).mean() - 1.0

def f_up_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0).rolling(60).std()
    dn = r.where(r < 0).rolling(60).std()
    return up / dn.replace(0, np.nan)

def f_crash_mag_60(df, s):
    r = df['close'].pct_change()
    return r.where(r < 0).rolling(60).mean().abs()

def f_downsidedev_term_20_60(df, s):
    r = df['close'].pct_change()
    neg = r.where(r < 0)
    dd20 = np.sqrt((neg ** 2).rolling(20).mean())
    dd60 = np.sqrt((neg ** 2).rolling(60).mean())
    return dd20 / dd60.replace(0, np.nan)

def f_vol_contraction_mom_20x60(df, s):
    r = df['close'].pct_change()
    c = df['close']
    mom = c / c.shift(20) - 1.0
    v20 = r.rolling(20).std()
    v60 = r.rolling(60).std()
    return mom * (v20 / v60.replace(0, np.nan) < 1.0).astype(float)

cands = {
    'trend_r2_60': (f_trend_r2_60, 'R^2 of 60d linear trend on log close', 'trend quality/consistency'),
    'sortino_60': (f_sortino_60, 'mean(r)/sqrt(mean(neg^2)) over 60d', 'downside risk-adjusted return'),
    'profit_factor_60': (f_profit_factor_60, 'sum(pos r)/|sum(neg r)| over 60d', 'return quality / profit factor'),
    'us10y_beta_60': (f_us10y_beta_60, 'beta of asset ret to dUS10Y over 60d', 'bond sensitivity'),
    'us10y_corr_60': (f_us10y_corr_60, 'corr(asset ret, dUS10Y) over 60d', 'bond-equity correlation regime'),
    'rates_spread_beta_60': (f_rates_spread_beta_60, 'beta to d(US10Y-CN10Y) over 60d', 'yield-spread sensitivity'),
    'idio_mom_60_20': (f_idio_mom_60_20, '60d cumulative residual ret vs SPX beta', 'idiosyncratic momentum'),
    'tail_ratio_60': (f_tail_ratio_60, 'p95(|up tail|)/p05(down tail) of daily rets 60d', 'tail asymmetry'),
    'range_vol_adj_mom_20': (f_range_vol_adj_mom_20, '20d mom / Parkinson vol(20)', 'range-based risk-adjusted momentum'),
    'close_ma60_pos': (f_close_ma60_pos, 'close/MA60 - 1', 'moving-average distance'),
    'up_vol_ratio_60': (f_up_vol_ratio_60, 'std(up days)/std(down days) 60d', 'up/down volatility asymmetry'),
    'crash_mag_60': (f_crash_mag_60, 'mean(|neg ret|) over 60d', 'downside crash severity'),
    'downsidedev_term_20_60': (f_downsidedev_term_20_60, 'downside dev 20d/60d', 'downside-risk term structure'),
    'vol_contraction_mom_20x60': (f_vol_contraction_mom_20x60, '20d mom x (vol20<vol60)', 'momentum x vol contraction'),
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
    recent = ic10[(gv >= pd.Timestamp('2025-07-15')) & (gv <= pd.Timestamp('2026-07-15'))]
    if np.isfinite(recent).sum() > 30:
        m['recent_1y_ic'] = float(np.nanmean(recent))
        sd = float(np.nanstd(recent, ddof=1))
        m['recent_1y_icir'] = float(np.nanmean(recent) / sd) if sd > 0 else 0.0
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
        if 'recent_1y_ic' in m:
            print(f"  recent_1y: ic={m['recent_1y_ic']:.4f} icir={m['recent_1y_icir']:.4f}", flush=True)
        print(f"  ADMISSION {'PASS' if ok else 'FAIL'} (|IC|={abs(m['ic']):.4f}/0.007 |ICIR|={abs(m['icir']):.4f}/0.084 rho={rho:.3f}/0.5)", flush=True)
    except Exception as e:
        print(f"{fid}: EXCEPTION {e}", flush=True)
        results[fid] = {'ok': False, 'error': str(e), 'desc': desc, 'tag': tag}

with open('scripts/miner_3_20261008_results_round23.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)

print("\n=== SUMMARY ===")
for fid, r in sorted(results.items()):
    if 'metrics' in r:
        m = r['metrics']
        print(f"{fid:26s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m.get('max_abs_library_correlation', float('nan')):.3f} ({m.get('max_corr_library_id')})")
    else:
        print(f"{fid:26s} ERROR {r.get('error', '')[:80]}")
print(f"total time {time.time()-t0:.1f}s")
