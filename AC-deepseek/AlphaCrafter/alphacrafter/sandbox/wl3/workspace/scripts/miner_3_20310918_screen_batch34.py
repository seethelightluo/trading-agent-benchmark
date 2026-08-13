"""miner_3 2031-09-18: screen batch-34 novel factor candidates on the 15-asset
cross-asset universe. Warm-up admission gates (2020-01-01..2026-07-15):
  |IC10| >= 0.007, |ICIR10| >= 0.084, library |rho| < 0.5 (gate re-checks).

Candidate families (novel vs library): close-location value (CLV), Kaufman trend
efficiency, return autocorrelation, Parkinson vol ratio, BTC beta, leave-one-out
global EW-basket beta, signed gap bias, downside-vol asymmetry, drawdown depth,
high/low asymmetry, rate-spread (CN10Y-US10Y) beta, SPX-residual momentum.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, WATCHLIST, VAL_START, VAL_END,
                           factor_to_panel, forward_returns)

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
# leave-one-out equal-weight basket return
ew_ex = {}
for s in WATCHLIST:
    others = [x for x in WATCHLIST if x != s]
    ew_ex[s] = ret_wide[others].mean(axis=1, skipna=True)

us10y = prices['US10Y']['close']
cn10y = prices['CN10Y']['close']
spread = cn10y - us10y  # yield spread level
spread_chg = spread.diff()
print(f"rate spread: {spread.dropna().index.min().date()}..{spread.dropna().index.max().date()} ({time.time()-t0:.1f}s)")


# ---------------- candidate factor functions ----------------
def f_clv_20(df, s):
    h = df['high']; l = df['low']; c = df['close']
    rng = (h - l).replace(0, np.nan)
    clv = (c - l) / rng
    return clv.rolling(20, min_periods=10).mean().reindex(df.index)


def f_kaufman_eff_20(df, s):
    c = df['close']
    r = c.pct_change()
    net = (c / c.shift(20) - 1.0).abs()
    path = r.abs().rolling(20, min_periods=10).sum()
    return (net / path.replace(0, np.nan)).reindex(df.index)


def f_ret_autocorr_20(df, s):
    r = df['close'].pct_change()
    ac = r.rolling(20, min_periods=10).apply(lambda y: pd.Series(y).autocorr(1), raw=False)
    return ac.reindex(df.index)


def f_parkinson_ratio_20(df, s):
    h = df['high']; l = df['low']; c = df['close']
    r = c.pct_change()
    iv = (np.log(h / l) ** 2).rolling(20, min_periods=10).mean() / (4.0 * np.log(2.0))
    cv = r.rolling(20, min_periods=10).var()
    return np.sqrt(iv / cv.replace(0, np.nan)).reindex(df.index)


def f_btc_beta_60(df, s):
    r = df['close'].pct_change()
    br = prices['BTC']['close'].pct_change()
    z = pd.concat([r.rename('r'), br.rename('b')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['b']) / z['b'].rolling(60, min_periods=30).var()
    return b.reindex(df.index)


def f_gbl_beta_60_ex(df, s):
    r = df['close'].pct_change()
    br = ew_ex[s]
    z = pd.concat([r.rename('r'), br.rename('b')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['b']) / z['b'].rolling(60, min_periods=30).var()
    return b.reindex(df.index)


def f_gap_signed_20(df, s):
    o = df['open']; c = df['close']
    gap = o / c.shift(1) - 1.0
    return gap.rolling(20, min_periods=10).mean().reindex(df.index)


def f_down_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    up = r.where(r > 0)
    dn = (-r).where(r < 0)
    return (dn.rolling(60, min_periods=30).std() /
            up.rolling(60, min_periods=30).std().replace(0, np.nan)).reindex(df.index)


def f_max_dd_60(df, s):
    c = df['close']
    return (c / c.rolling(60, min_periods=30).max() - 1.0).reindex(df.index)


def f_hilo_asym_60(df, s):
    h = df['high']; l = df['low']; c = df['close']
    up = h.rolling(60, min_periods=30).max() - c
    dn = c - l.rolling(60, min_periods=30).min()
    return (up / dn.replace(0, np.nan)).reindex(df.index)


def f_rate_spread_beta_60(df, s):
    r = df['close'].pct_change()
    sc = spread_chg
    z = pd.concat([r.rename('r'), sc.rename('s')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['s']) / z['s'].rolling(60, min_periods=30).var()
    return b.reindex(df.index)


def f_resid_mom_20(df, s):
    c = df['close']
    r = c.pct_change()
    spx = prices['SPX']['close'].pct_change()
    z = pd.concat([r.rename('r'), spx.rename('s')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['s']) / z['s'].rolling(60, min_periods=30).var()
    exp = b * spx
    resid = (r - exp).rolling(20, min_periods=10).sum()
    return resid.reindex(df.index)


candidates = {
    'clv_20': (f_clv_20, 'mean((close-low)/(high-low)) over 20d (accumulation/distribution)'),
    'kaufman_eff_20': (f_kaufman_eff_20, '|20d net move| / sum(|daily ret|,20) (trend efficiency)'),
    'ret_autocorr_20': (f_ret_autocorr_20, 'lag-1 autocorrelation of daily returns over 20d'),
    'parkinson_ratio_20': (f_parkinson_ratio_20, 'sqrt(mean(ln(H/L)^2)/(4ln2) / var(close ret)) over 20d'),
    'btc_beta_60': (f_btc_beta_60, 'rolling beta of asset ret on BTC ret, 60d (crypto beta)'),
    'gbl_beta_60_ex': (f_gbl_beta_60_ex, 'rolling beta of asset ret on leave-one-out EW 14-asset basket, 60d'),
    'gap_signed_20': (f_gap_signed_20, 'mean(open/prev_close - 1) over 20d (signed gap bias)'),
    'down_vol_ratio_60': (f_down_vol_ratio_60, 'std(down rets)/std(up rets) over 60d (vol asymmetry)'),
    'max_dd_60': (f_max_dd_60, 'close/rolling_max(close,60) - 1 (drawdown depth)'),
    'hilo_asym_60': (f_hilo_asym_60, '(max_high_60 - close)/(close - min_low_60) (range asymmetry)'),
    'rate_spread_beta_60': (f_rate_spread_beta_60, 'rolling beta of asset ret on d(CN10Y-US10Y), 60d'),
    'resid_mom_20': (f_resid_mom_20, '20d sum of SPX-beta-residual daily ret (idiosyncratic momentum)'),
}

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
    panel = factor_to_panel(fn, prices)
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

out = Path('scripts/miner_3_20310918_results_batch34.json')
out.write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
print(f"\nresults saved to {out} ({time.time()-t0:.1f}s total)")
