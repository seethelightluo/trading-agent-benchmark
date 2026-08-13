"""miner_3 2032-04-01: screen batch-35 novel factor candidates on the 15-asset
cross-asset universe. Warm-up admission gates (2020-01-01..2026-07-15):
  |IC10| >= 0.007, |ICIR10| >= 0.084, library |rho| < 0.5 (deterministic gate re-checks).

Candidate families (novel vs library + batches 33/34):
  var_ratio_20_5      - variance ratio (long-memory: trending vs mean-reverting regime)
  trend_tstat_60      - OLS log-price slope t-stat over 60d (trend strength w/ noise control)
  above_ma_ratio_60   - fraction of days close > MA60 (trend breadth)
  ma_cross_persist_20 - persistence of sign(MA20-MA60) over 20d (MA-crossover regime)
  gap_reversal_20     - mean(sign(gap) * intraday ret) (gap follow-through vs reversal)
  range_slope_20      - 20d return / mean daily range (trend per unit range)
  corr_ewbasket_60    - rolling corr with leave-one-out EW 14-asset basket (co-movement level)
  lag_spx_beta_60     - rolling beta of asset ret on LAGGED SPX ret (lead-lag spillover)
  us10y_beta_60       - rolling beta on US10Y yield change (US rate sensitivity)
  beta_stability_60   - std of 20d rolling SPX-beta over 60d (market-regime instability)
  er_ratio_10_60      - ER(10)/ER(60) trend-efficiency acceleration
  overnight_bias_20   - mean overnight ret minus mean intraday ret (overnight dominance)
  ndx_beta_60         - rolling beta on NDX (tech beta)
  co_movement_20      - same-direction breadth vs other 14 assets (herding vs contrarian)
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import load_prices, load_index, WATCHLIST, VAL_START, VAL_END

t0 = time.time()
prices = load_prices(days=3200)
print(f"assets loaded: {len(prices)}; max date: {max(d.index.max() for d in prices.values())} ({time.time()-t0:.1f}s)")

idx = set()
for s, df in prices.items():
    idx.update(df.index)
gidx = pd.DatetimeIndex(sorted(idx))
gidx = gidx[gidx >= VAL_START]
print(f"trading grid: {len(gidx)} dates, {gidx.min().date()}..{gidx.max().date()}")

cal_grid = pd.date_range(VAL_START, VAL_END, freq='D')
print(f"library calendar grid: {len(cal_grid)} dates ({time.time()-t0:.1f}s)")


def row_spearman(X, Y, min_valid=8):
    X = pd.DataFrame(X, dtype=float)
    Y = pd.DataFrame(Y, dtype=float)
    m = X.notna() & Y.notna()
    n = m.sum(axis=1)
    X2 = X.where(m); Y2 = Y.where(m)
    rx = X2.rank(axis=1)
    ry = Y2.rank(axis=1)
    rxm = rx.sub(rx.mean(axis=1), axis=0)
    rym = ry.sub(ry.mean(axis=1), axis=0)
    num = (rxm * rym).sum(axis=1)
    den = np.sqrt((rxm ** 2).sum(axis=1) * (rym ** 2).sum(axis=1))
    rho = (num / den.replace(0, np.nan)).to_numpy(dtype=float).copy()
    rho[n < min_valid] = np.nan
    return rho


# ---- precompute cross-asset reference series ----
ret_wide = pd.DataFrame({s: prices[s]['close'].pct_change() for s in WATCHLIST}).sort_index()
ew_ex = {}
for s in WATCHLIST:
    others = [x for x in WATCHLIST if x != s]
    ew_ex[s] = ret_wide[others].mean(axis=1, skipna=True)
print(f"basket refs built ({time.time()-t0:.1f}s)")


# ---------------- candidate factor functions ----------------
def f_var_ratio_20_5(df, s):
    r = df['close'].pct_change()
    v1 = r.rolling(20, min_periods=10).var()
    r5 = r.rolling(5).sum()
    v5 = r5.rolling(20, min_periods=10).var()
    return (v5 / (5.0 * v1).replace(0, np.nan)).reindex(df.index)


def f_trend_tstat_60(df, s):
    y = np.log(df['close'])
    w = 60
    x = np.arange(w, dtype=float)
    xm = x - x.mean()
    sxx = float((xm ** 2).sum())

    def _t(v):
        if np.any(~np.isfinite(v)):
            return np.nan
        ym = v.mean()
        b = float(np.dot(xm, v - ym)) / sxx
        resid = (v - ym) - b * xm
        sse = float(np.dot(resid, resid))
        se = np.sqrt(sse / (w - 2.0) / sxx)
        return b / se
    return y.rolling(w, min_periods=w).apply(_t, raw=True).reindex(df.index)


def f_above_ma_ratio_60(df, s):
    c = df['close']
    ma = c.rolling(60, min_periods=30).mean()
    above = (c > ma).astype(float)
    return above.rolling(60, min_periods=30).mean().reindex(df.index)


def f_ma_cross_persist_20(df, s):
    c = df['close']
    sgn = np.sign(c.rolling(20, min_periods=10).mean() - c.rolling(60, min_periods=30).mean())
    return sgn.rolling(20, min_periods=10).mean().reindex(df.index)


def f_gap_reversal_20(df, s):
    o = df['open']; c = df['close']
    gap = o / c.shift(1) - 1.0
    intr = c / o - 1.0
    score = np.sign(gap) * intr
    return score.rolling(20, min_periods=10).mean().reindex(df.index)


def f_range_slope_20(df, s):
    h = df['high']; l = df['low']; c = df['close']
    m = c / c.shift(20) - 1.0
    rng = (h - l).rolling(20, min_periods=10).mean()
    return (m / rng.replace(0, np.nan)).reindex(df.index)


def f_corr_ewbasket_60(df, s):
    r = df['close'].pct_change()
    br = ew_ex[s]
    z = pd.concat([r.rename('r'), br.rename('b')], axis=1).dropna()
    c = z['r'].rolling(60, min_periods=30).corr(z['b'])
    return c.reindex(df.index)


def f_lag_spx_beta_60(df, s):
    r = df['close'].pct_change()
    lr = prices['SPX']['close'].pct_change().shift(1)
    z = pd.concat([r.rename('r'), lr.rename('l')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['l']) / z['l'].rolling(60, min_periods=30).var()
    return b.reindex(df.index)


def f_us10y_beta_60(df, s):
    r = df['close'].pct_change()
    yr = prices['US10Y']['close'].pct_change()
    z = pd.concat([r.rename('r'), yr.rename('y')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['y']) / z['y'].rolling(60, min_periods=30).var()
    return b.reindex(df.index)


def f_beta_stability_60(df, s):
    r = df['close'].pct_change()
    sr = prices['SPX']['close'].pct_change()
    z = pd.concat([r.rename('r'), sr.rename('s')], axis=1).dropna()
    b = z['r'].rolling(20, min_periods=10).cov(z['s']) / z['s'].rolling(20, min_periods=10).var()
    return b.rolling(60, min_periods=30).std().reindex(df.index)


def f_er_ratio_10_60(df, s):
    c = df['close']; r = c.pct_change()
    er10 = (c / c.shift(10) - 1.0).abs() / r.abs().rolling(10, min_periods=5).sum().replace(0, np.nan)
    er60 = (c / c.shift(60) - 1.0).abs() / r.abs().rolling(60, min_periods=30).sum().replace(0, np.nan)
    return (er10 / er60.replace(0, np.nan)).reindex(df.index)


def f_overnight_bias_20(df, s):
    o = df['open']; c = df['close']
    on = o / c.shift(1) - 1.0
    intr = c / o - 1.0
    return (on.rolling(20, min_periods=10).mean() -
            intr.rolling(20, min_periods=10).mean()).reindex(df.index)


def f_ndx_beta_60(df, s):
    r = df['close'].pct_change()
    nr = prices['NDX']['close'].pct_change()
    z = pd.concat([r.rename('r'), nr.rename('n')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['n']) / z['n'].rolling(60, min_periods=30).var()
    return b.reindex(df.index)


def f_co_movement_20(df, s):
    m = ret_wide[s].rolling(20, min_periods=10).sum()
    others = [x for x in WATCHLIST if x != s]
    om = ret_wide[others].rolling(20, min_periods=10).sum()
    same = (om.mul(np.sign(m), axis=0) > 0).astype(float)
    return same.mean(axis=1).reindex(df.index)


candidates = {
    'var_ratio_20_5': (f_var_ratio_20_5, 'Var(5d sum ret)/(5*Var(1d ret)) over 20d (long-memory)'),
    'trend_tstat_60': (f_trend_tstat_60, 'OLS log-price slope t-stat over 60d (trend strength)'),
    'above_ma_ratio_60': (f_above_ma_ratio_60, 'fraction of 60d with close > MA60 (trend breadth)'),
    'ma_cross_persist_20': (f_ma_cross_persist_20, 'mean sign(MA20-MA60) over 20d (MA regime)'),
    'gap_reversal_20': (f_gap_reversal_20, 'mean(sign(gap)*intraday ret) 20d (gap follow-through)'),
    'range_slope_20': (f_range_slope_20, '20d return / mean daily range (trend per unit range)'),
    'corr_ewbasket_60': (f_corr_ewbasket_60, '60d corr with leave-one-out EW basket'),
    'lag_spx_beta_60': (f_lag_spx_beta_60, '60d beta on 1-day lagged SPX ret (lead-lag)'),
    'us10y_beta_60': (f_us10y_beta_60, '60d beta on US10Y yield change (US rate sensitivity)'),
    'beta_stability_60': (f_beta_stability_60, 'std of 20d SPX-beta over 60d (beta instability)'),
    'er_ratio_10_60': (f_er_ratio_10_60, 'ER(10)/ER(60) trend-efficiency acceleration'),
    'overnight_bias_20': (f_overnight_bias_20, 'mean(overnight ret)-mean(intraday ret) 20d'),
    'ndx_beta_60': (f_ndx_beta_60, '60d beta on NDX ret (tech beta)'),
    'co_movement_20': (f_co_movement_20, 'same-direction breadth vs other 14 assets 20d'),
}

from factor_common import forward_returns
fwd_mats = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd = forward_returns(prices, h).reindex(gidx)
    fwd_mats[h] = fwd[WATCHLIST].values.astype(float)

# library artifacts with per-factor grids from JSON metadata
lib_artifacts = {}
for jp in sorted(Path('factors').glob('*.json')):
    try:
        p = json.loads(jp.read_text(encoding='utf-8'))
        art = p.get('signal_artifact')
        if not art:
            continue
        arr = np.load(Path('factors') / art, allow_pickle=False)
        if arr.ndim != 2 or arr.shape[1] != 15:
            continue
        g = p.get('signal_artifact_grid', {})
        grid = None
        try:
            cand = pd.date_range(pd.Timestamp(g['start']), pd.Timestamp(g['end']), freq='D')
            if len(cand) == g.get('n_dates') and len(cand) == arr.shape[0]:
                grid = cand
        except Exception:
            pass
        if grid is None:
            if arr.shape[0] == len(cal_grid):
                grid = cal_grid
        if grid is not None:
            lib_artifacts[p.get('factor_id', jp.stem)] = (grid, arr)
    except Exception:
        pass
print(f"library artifacts with usable grids: {len(lib_artifacts)} ({time.time()-t0:.1f}s)")


def max_lib_corr(panel):
    best, best_id = 0.0, None
    for fid, (grid, la) in lib_artifacts.items():
        mc = panel.reindex(grid)[WATCHLIST].values.astype(float)
        c = row_spearman(mc, la)
        c = c[np.isfinite(c)]
        if len(c):
            r = float(np.abs(c).mean())
            if r > best:
                best, best_id = r, fid
    return best, best_id


warm = (gidx >= VAL_START) & (gidx <= VAL_END)
rstart = VAL_END + pd.Timedelta(days=1)
recent = gidx >= rstart
recent = recent & (gidx <= gidx.max() - pd.Timedelta(days=15))

results = {}
for fid, (fn, desc) in candidates.items():
    t1 = time.time()
    panel = pd.DataFrame({s: fn(prices[s], s) for s in WATCHLIST}).sort_index()
    panel = panel[~panel.index.duplicated(keep='last')]
    if panel.empty:
        print(f"{fid}: EMPTY panel"); continue
    mat = panel.reindex(gidx)[WATCHLIST].values.astype(float)
    ics = {}
    for h in (1, 2, 3, 5, 10, 20):
        ics[h] = row_spearman(mat, fwd_mats[h])
    ic10w = ics[10][warm]
    ic10w = ic10w[np.isfinite(ic10w)]
    if len(ic10w) < 100:
        print(f"{fid}: insufficient warm IC dates {len(ic10w)}"); continue
    ic = float(ic10w.mean()); sd = float(ic10w.std(ddof=1))
    icir = ic / sd if sd > 0 else 0.0
    hit = float((ic10w > 0).mean()) if ic >= 0 else float((ic10w < 0).mean())
    fac = panel[(panel.index >= VAL_START) & (panel.index <= VAL_END)]
    cov = float(fac.notna().sum().sum()) / (fac.shape[0] * fac.shape[1]) if fac.shape[0] else 0.0
    ge8 = float((fac.notna().sum(axis=1) >= 8).mean())
    turn = float(fac.rank(axis=1).diff(10).abs().mean().mean()) if len(fac) > 10 else float('nan')
    decay = {str(h): float(np.nanmean(ics[h][warm])) for h in (1, 2, 3, 5, 10, 20)}
    rho, rho_id = max_lib_corr(panel)
    icr = ics[10][recent]
    icr = icr[np.isfinite(icr)]
    ic_rmean = float(icr.mean()) if len(icr) >= 30 else float('nan')
    ic_rsd = float(icr.std(ddof=1)) if len(icr) >= 30 else float('nan')
    ic_ricir = ic_rmean / ic_rsd if len(icr) >= 30 and ic_rsd > 0 else float('nan')
    ok = abs(ic) >= 0.007 and abs(icir) >= 0.084 and rho < 0.5
    results[fid] = {
        'desc': desc, 'ic': ic, 'icir': icir, 'hit': hit, 'cov': cov, 'ge8': ge8,
        'turn': turn, 'decay': decay, 'rho': rho, 'rho_id': rho_id,
        'ic_recent': ic_rmean, 'icir_recent': ic_ricir, 'n_recent': int(len(icr)),
        'n_warm': int(len(ic10w)), 'PASS': ok,
    }
    print(f"\n=== {fid} | {desc} | {time.time()-t1:.1f}s ===")
    print(f"  IC10={ic:+.4f} ICIR10={icir:+.4f} hit={hit:.3f} cov={cov:.3f} ge8={ge8:.3f} turn={turn:.3f}")
    print(f"  decay(1,2,3,5,10,20)={[round(decay[str(h)],4) for h in (1,2,3,5,10,20)]}")
    print(f"  max_lib_rho={rho:.3f} ({rho_id}) | recent_IC={ic_rmean:+.4f} recent_ICIR={ic_ricir:+.4f} n={len(icr)}")
    print(f"  ADMISSION: |IC|={abs(ic):.4f}>=0.007 {abs(ic)>=0.007} | |ICIR|={abs(icir):.4f}>=0.084 {abs(icir)>=0.084} | rho<0.5 {rho<0.5} -> {'PASS' if ok else 'FAIL'}")

out = Path('scripts/miner_3_20320401_results_batch35.json')
out.write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
print(f"\nresults saved to {out} ({time.time()-t0:.1f}s total)")
