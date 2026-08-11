"""Round 22 (2026-10-08) fast numpy screen for novel factors on the 15-asset universe.

Same 14 candidates as the (too slow) pandas version, but with vectorized
per-date Spearman implementations. Gate: |IC10|>=0.007, |ICIR10|>=0.084,
max_abs_library_correlation < 0.5 vs the 18 effective library artifacts.
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
T = len(grid)
N = len(WATCHLIST)
print(f"grid {T} dates {grid.min().date()}..{grid.max().date()} | assets {len(prices)} | {time.time()-t0:.1f}s", flush=True)

# ---------------- library rank matrices from real signal artifacts ----------------
lib_ranks = {}
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
            lib_ranks[fid] = arr
    except Exception as e:
        print(f"  artifact ERR {f}: {e}", flush=True)
print(f"library artifacts: {len(lib_ranks)} -> {sorted(lib_ranks.keys())}", flush=True)


def row_rank(mat):
    """Row-wise mean ranks over valid entries (1..k), NaN preserved. (T,N)."""
    valid = np.isfinite(mat)
    tmp = np.where(valid, mat, np.inf)
    order = np.argsort(tmp, axis=1, kind='mergesort')
    pos = np.empty_like(mat, dtype=float)
    pos[np.arange(mat.shape[0])[:, None], order] = np.arange(1, mat.shape[1] + 1)
    pos = np.where(valid, pos, np.nan)
    # average ranks for ties (within row)
    smat = np.where(valid, mat, np.nan)
    out = pos.copy()
    for i in range(mat.shape[0]):
        row = smat[i]
        m = np.isfinite(row)
        if m.sum() < 2:
            continue
        uvals, inv, cnt = np.unique(row[m], return_inverse=True, return_counts=True)
        if (cnt > 1).any():
            avg = np.zeros(len(uvals))
            for u in range(len(uvals)):
                avg[u] = pos[i][m][inv == u].mean()
            out[i][m] = avg[inv]
    return out


def spearman_mat(xm, ym):
    """Per-date Spearman between two (T,N) rank matrices (NaN-aware)."""
    xr = row_rank(xm)
    yr = row_rank(ym)
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


def max_lib_corr(panel):
    pm = signal_matrix(panel, grid)          # (T,15) raw values
    best, best_id, per_id = 0.0, None, {}
    for fid, lm in lib_ranks.items():
        r = spearman_mat(pm, lm)
        rr = r[np.isfinite(r)]
        if len(rr) > 0:
            per_id[fid] = float(np.mean(rr))
            if abs(per_id[fid]) > best:
                best, best_id = abs(per_id[fid]), fid
    return best, best_id, per_id

# ---------------- macro / market inputs ----------------
spx_ret = prices['SPX']['close'].pct_change()
ndx_ret = prices['NDX']['close'].pct_change()
btc_ret = prices['BTC']['close'].pct_change()
wti_ret = prices['WTI']['close'].pct_change()
us10y = prices['US10Y']['close']
jpy = load_index('USDJPY', prices=prices)


def _cond_beta(r, m, cond, w):
    z = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    z = z[cond(z['m'])]
    if len(z) < w + 5:
        return pd.Series(np.nan, index=r.index)
    b = z['r'].rolling(w).cov(z['m']) / z['m'].rolling(w).var().replace(0, np.nan)
    return b.reindex(r.index)


def f_roll_kurt_60(df, s):
    return df['close'].pct_change().rolling(60).kurt()


def f_jump_intensity_60(df, s):
    r = df['close'].pct_change()
    sd = r.rolling(20).std()
    ind = (r.abs() > 2.0 * sd).astype(float)
    return ind.rolling(60).mean()


def f_us10y_corr_60(df, s):
    r = df['close'].pct_change()
    dy = us10y.diff()
    z = pd.concat([r.rename('r'), dy.rename('y')], axis=1).dropna()
    return z['r'].rolling(60).corr(z['y']).reindex(z.index)


def f_btc_down_beta(df, s):
    r = df['close'].pct_change()
    b = _cond_beta(r, btc_ret, lambda m: m < 0, 60)
    return (-b * (btc_ret.rolling(20).mean() * 20)).reindex(r.index)


def f_weekday_effect_120(df, s):
    r = df['close'].pct_change()
    dow = r.index.dayofweek
    mon = r.where(dow == 0).rolling(120, min_periods=20).mean()
    fri = r.where(dow == 4).rolling(120, min_periods=20).mean()
    return (mon - fri)


def f_tom_effect_120(df, s):
    r = df['close'].pct_change()
    dom = r.index.day
    early = r.where(dom <= 3).rolling(120, min_periods=15).mean()
    late = r.where(dom >= 26).rolling(120, min_periods=15).mean()
    return (early - late)


def f_overnight_intraday_corr_20(df, s):
    gap = df['open'] / df['close'].shift(1) - 1.0
    intra = df['close'] / df['open'] - 1.0
    z = pd.concat([gap.rename('g'), intra.rename('i')], axis=1).dropna()
    return z['g'].rolling(20).corr(z['i']).reindex(z.index)


def f_trend_r2_60(df, s):
    c = df['close']
    idx = pd.Series(np.arange(len(c)), index=c.index)
    corr = c.rolling(60).corr(idx)
    return (corr ** 2)


def f_dd_depth_60(df, s):
    return df['close'] / df['close'].rolling(60).max() - 1.0


def f_tech_beta_diff_60(df, s):
    r = df['close'].pct_change()
    b_ndx = r.rolling(60).cov(ndx_ret) / ndx_ret.rolling(60).var().replace(0, np.nan)
    b_spx = r.rolling(60).cov(spx_ret) / spx_ret.rolling(60).var().replace(0, np.nan)
    return (b_ndx - b_spx)


def f_wti_beta_cond(df, s):
    r = df['close'].pct_change()
    b = _cond_beta(r, wti_ret, lambda m: m > 0, 60)
    return (b * (wti_ret.rolling(20).mean() * 20)).reindex(r.index)


def f_jpy_beta_cond(df, s):
    r = df['close'].pct_change()
    jr = jpy['close'].pct_change()
    b = _cond_beta(r, jr, lambda m: m < 0, 60)
    return (b * (jr.rolling(20).mean() * 20)).reindex(r.index)


def f_vol_skew_20_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(20).skew() - r.rolling(60).skew()


def f_autocorr_abs_ret_20(df, s):
    ar = df['close'].pct_change().abs()
    return ar.rolling(20).corr(ar.shift(1))


cands = {
    'roll_kurt_60': (f_roll_kurt_60, 'rolling 60d kurtosis of daily returns', 'higher-moment fat-tail risk'),
    'jump_intensity_60': (f_jump_intensity_60, 'share of days with |r|>2*std20 in 60d', 'jump frequency / tail activity'),
    'us10y_corr_60': (f_us10y_corr_60, 'rolling 60d corr(asset ret, dUS10Y)', 'bond-equity correlation regime'),
    'btc_down_beta_60x20': (f_btc_down_beta, '-downside beta on BTC x BTC 20d trend', 'crypto tail sensitivity'),
    'weekday_effect_120': (f_weekday_effect_120, 'mean Monday minus Friday return (120d)', 'calendar weekday seasonality'),
    'tom_effect_120': (f_tom_effect_120, 'mean early-month minus late-month return (120d)', 'turn-of-month calendar effect'),
    'overnight_intraday_corr_20': (f_overnight_intraday_corr_20, 'corr(overnight gap, intraday ret) over 20d', 'within-day gap continuation/reversal'),
    'trend_r2_60': (f_trend_r2_60, 'R^2 of 60d linear trend on close', 'trend quality/consistency'),
    'dd_depth_60': (f_dd_depth_60, 'close/rolling_max(60)-1 (drawdown depth)', 'drawdown depth risk'),
    'tech_beta_diff_60': (f_tech_beta_diff_60, 'beta(NDX,60) - beta(SPX,60)', 'relative tech sensitivity'),
    'wti_beta_cond_60x20': (f_wti_beta_cond, 'beta on WTI on WTI-up days x WTI 20d trend', 'oil sensitivity conditional'),
    'jpy_beta_cond_60x20': (f_jpy_beta_cond, 'beta on USDJPY on JPY-up days x USDJPY 20d trend', 'JPY carry sensitivity'),
    'vol_skew_20_60': (f_vol_skew_20_60, 'skew(r,20)-skew(r,60)', 'skew term structure'),
    'autocorr_abs_ret_20': (f_autocorr_abs_ret_20, 'lag-1 autocorr of |r| over 20d', 'volatility clustering persistence'),
}

# precompute forward-return rank matrices (vectorized)
fwd_raw = {h: forward_returns(prices, h) for h in (1, 2, 3, 5, 10, 20)}
fwd_rank = {h: signal_matrix(fwd_raw[h].reindex(grid), grid) for h in fwd_raw}
fwd_rank = {h: row_rank(m) for h, m in fwd_rank.items()}

val_mask = (grid >= VAL_START) & (grid <= VAL_END)


def fast_validate(fid, panel):
    pm = signal_matrix(panel, grid)
    pr = row_rank(pm)
    ic10 = spearman_mat(pr, fwd_rank[10])
    ic10 = ic10[val_mask]
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
        s = spearman_mat(pr, fwd_rank[h])[val_mask]
        decay[str(h)] = float(np.nanmean(s))
    m = {'ic': ic_mean, 'icir': icir, 'ic_hit_ratio': hit,
         'n_ic_dates': int(np.isfinite(ic10).sum()), 'coverage_asset_days': coverage,
         'coverage_dates_ge8': ge8, 'turnover_10d_rank': turn,
         'decay_ic_by_horizon': decay}
    # sub-periods
    for nm, a, b in [('ic_2020_2022', '2020-01-01', '2022-12-31'),
                     ('ic_2023_2024', '2023-01-01', '2024-12-31'),
                     ('ic_2025_2026', '2025-01-01', '2026-07-15')]:
        sub = ic10[(grid[val_mask] >= pd.Timestamp(a)) & (grid[val_mask] <= pd.Timestamp(b))]
        m[nm] = float(np.nanmean(sub)) if np.isfinite(sub).sum() > 30 else float('nan')
    recent = ic10[(grid[val_mask] >= pd.Timestamp('2025-07-15')) & (grid[val_mask] <= pd.Timestamp('2026-07-15'))]
    if np.isfinite(recent).sum() > 30:
        m['recent_1y_ic'] = float(np.nanmean(recent))
        sd = float(np.nanstd(recent, ddof=1))
        m['recent_1y_icir'] = float(np.nanmean(recent) / sd) if sd > 0 else 0.0
    return m


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

with open('scripts/miner_3_20261008_results_round22.json', 'w') as fh:
    json.dump(results, fh, indent=1, default=str)

print("\n=== SUMMARY ===")
for fid, r in sorted(results.items()):
    if 'metrics' in r:
        m = r['metrics']
        print(f"{fid:26s} ok={r['ok']} ic={m['ic']:.4f} icir={m['icir']:.4f} rho={m.get('max_abs_library_correlation', float('nan')):.3f} ({m.get('max_corr_library_id')})")
    else:
        print(f"{fid:26s} ERROR {r.get('error', '')[:80]}")
print(f"total time {time.time()-t0:.1f}s")
