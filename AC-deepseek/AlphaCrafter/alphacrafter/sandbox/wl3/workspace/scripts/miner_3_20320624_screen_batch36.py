"""miner_3 2032-06-24: screen batch-36 novel factor candidates on the 15-asset
cross-asset universe. Warm-up admission gates (2020-01-01..2026-07-15):
  |IC10| >= 0.007, |ICIR10| >= 0.084, library |rho| < 0.5 (deterministic gate re-checks).

Candidate families (novel vs library + batches 32-35; emphasis on the under-used
volume dimension, cross-sectional dispersion, and regime-position clocks):
  vol_confirm_20       - 20d corr(ret, vol/vol_ma60): volume-confirmed price moves
  vol_surprise_5_60    - (vol_ma5 - vol_ma60)/std60(vol): volume expansion z-score
  obv_norm_slope_20    - OBV 20d slope normalized by OBV-change std (volume flow)
  vwap_dist_20         - (close - vwap20)/mean_range20: distance from VWAP
  updown_vol_asym_20   - (down_vol - up_vol)/(down_vol+up_vol): signed vol asymmetry
  xsec_rs_disp_20      - 20d ret / cross-sectional std of 20d rets (RS per dispersion)
  gap_streak_20        - mean consecutive same-sign gap length (gap persistence)
  trend_snr_20_60      - |ret20|/(std60*sqrt(20)): trend signal-to-noise
  wti_beta_60          - 60d beta on WTI returns (energy beta)
  xau_beta_60          - 60d beta on XAU returns (gold beta)
  dd_recovery_120      - time since 120d high / 120 (drawdown recovery clock)
  skew_stability_60    - std of 20d rolling skew over 60d (skew regime stability)
  vol_percentile_20_120- percentile of 20d vol within trailing 120d (vol regime pos)
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import load_prices, WATCHLIST, VAL_START, VAL_END

t0 = time.time()
prices = load_prices(days=3400)
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
xsec_std_20 = ret_wide.rolling(20, min_periods=10).std().mean(axis=1)  # avg asset vol
print(f"basket refs built ({time.time()-t0:.1f}s)")


# ---------------- candidate factor functions ----------------
def f_vol_confirm_20(df, s):
    r = df['close'].pct_change()
    v = df['volume'].astype(float)
    vma = v.rolling(60, min_periods=30).mean()
    vr = (v / vma.replace(0, np.nan)).clip(0, 10)
    z = pd.concat([r.rename('r'), vr.rename('v')], axis=1)
    c = z['r'].rolling(20, min_periods=10).corr(z['v'])
    return c.reindex(df.index)


def f_vol_surprise_5_60(df, s):
    v = df['volume'].astype(float)
    vma5 = v.rolling(5, min_periods=3).mean()
    vma60 = v.rolling(60, min_periods=30).mean()
    vsd = v.rolling(60, min_periods=30).std()
    return ((vma5 - vma60) / vsd.replace(0, np.nan)).reindex(df.index)


def f_obv_norm_slope_20(df, s):
    c = df['close']; v = df['volume'].astype(float)
    obv = (np.sign(c.diff()) * v).fillna(0.0).cumsum()
    d = obv.diff()
    sd = d.rolling(20, min_periods=10).std()
    slope = obv.diff(20)
    return (slope / (sd * 20.0).replace(0, np.nan)).reindex(df.index)


def f_vwap_dist_20(df, s):
    c = df['close']; h = df['high']; l = df['low']; v = df['volume'].astype(float)
    tp = (h + l + c) / 3.0
    vwap = (tp * v).rolling(20, min_periods=10).sum() / v.rolling(20, min_periods=10).sum().replace(0, np.nan)
    rng = (h - l).rolling(20, min_periods=10).mean()
    return ((c - vwap) / rng.replace(0, np.nan)).reindex(df.index)


def f_updown_vol_asym_20(df, s):
    r = df['close'].pct_change()
    up = r.clip(lower=0)
    dn = (-r).clip(lower=0)
    us = up.rolling(20, min_periods=10).std()
    ds = dn.rolling(20, min_periods=10).std()
    return ((ds - us) / (ds + us).replace(0, np.nan)).reindex(df.index)


def f_xsec_rs_disp_20(df, s):
    m = ret_wide[s].rolling(20, min_periods=10).sum()
    disp = ret_wide.rolling(20, min_periods=10).std().std(axis=1)  # xsec dispersion of vols
    return (m / disp.replace(0, np.nan)).reindex(df.index)


def f_gap_streak_20(df, s):
    o = df['open']; c = df['close']
    gap = np.sign(o / c.shift(1) - 1.0).fillna(0.0)
    # consecutive same-sign gap persistence: mean abs streak length via run encoding
    def _streak(x):
        out = np.full(len(x), np.nan)
        i = 0
        while i < len(x):
            j = i
            while j + 1 < len(x) and x[j + 1] == x[i] and x[i] != 0:
                j += 1
            if x[i] != 0:
                ln = j - i + 1
                out[i:j + 1] = ln
            i = j + 1
        return out
    streak = gap.rolling(20, min_periods=10).apply(lambda w: np.nanmean(_streak(w.to_numpy())), raw=False)
    return streak.reindex(df.index)


def f_trend_snr_20_60(df, s):
    c = df['close']
    r = c.pct_change()
    m = c / c.shift(20) - 1.0
    sd = r.rolling(60, min_periods=30).std()
    return (m.abs() / (sd * np.sqrt(20.0)).replace(0, np.nan)).reindex(df.index)


def f_wti_beta_60(df, s):
    if s == 'WTI':
        return pd.Series(np.nan, index=df.index)
    r = df['close'].pct_change()
    rr = prices['WTI']['close'].pct_change()
    z = pd.concat([r.rename('r'), rr.rename('w')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['w']) / z['w'].rolling(60, min_periods=30).var()
    return b.reindex(df.index)


def f_xau_beta_60(df, s):
    if s == 'XAU':
        return pd.Series(np.nan, index=df.index)
    r = df['close'].pct_change()
    rr = prices['XAU']['close'].pct_change()
    z = pd.concat([r.rename('r'), rr.rename('g')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['g']) / z['g'].rolling(60, min_periods=30).var()
    return b.reindex(df.index)


def f_dd_recovery_120(df, s):
    c = df['close']
    rollmax = c.rolling(120, min_periods=60).max()
    days_since_high = c.rolling(120, min_periods=60).apply(
        lambda w: int(np.argmax(w[::-1])) if len(w) else np.nan, raw=True)
    return (days_since_high / 120.0).reindex(df.index)


def f_skew_stability_60(df, s):
    r = df['close'].pct_change()
    sk = r.rolling(20, min_periods=10).skew()
    return sk.rolling(60, min_periods=30).std().reindex(df.index)


def f_vol_percentile_20_120(df, s):
    r = df['close'].pct_change()
    v20 = r.rolling(20, min_periods=10).std()
    lo = v20.rolling(120, min_periods=60).min()
    hi = v20.rolling(120, min_periods=60).max()
    return ((v20 - lo) / (hi - lo).replace(0, np.nan)).reindex(df.index)


candidates = {
    'vol_confirm_20': (f_vol_confirm_20, '20d corr(ret, vol/vol_ma60) volume-confirmed moves'),
    'vol_surprise_5_60': (f_vol_surprise_5_60, '(vol_ma5-vol_ma60)/std60(vol) volume expansion z'),
    'obv_norm_slope_20': (f_obv_norm_slope_20, 'OBV 20d slope / (20*std of daily OBV change)'),
    'vwap_dist_20': (f_vwap_dist_20, '(close - vwap20)/mean_range20 distance from VWAP'),
    'updown_vol_asym_20': (f_updown_vol_asym_20, '(down_vol-up_vol)/(down_vol+up_vol) vol asymmetry'),
    'xsec_rs_disp_20': (f_xsec_rs_disp_20, '20d ret / xsec std of 20d vols (RS per dispersion)'),
    'gap_streak_20': (f_gap_streak_20, 'mean consecutive same-sign gap length 20d'),
    'trend_snr_20_60': (f_trend_snr_20_60, '|ret20|/(std60*sqrt(20)) trend signal-to-noise'),
    'wti_beta_60': (f_wti_beta_60, '60d beta on WTI returns (energy beta)'),
    'xau_beta_60': (f_xau_beta_60, '60d beta on XAU returns (gold beta)'),
    'dd_recovery_120': (f_dd_recovery_120, 'time since 120d high / 120 (recovery clock)'),
    'skew_stability_60': (f_skew_stability_60, 'std of 20d rolling skew over 60d'),
    'vol_percentile_20_120': (f_vol_percentile_20_120, 'percentile of 20d vol within 120d'),
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
    try:
        panel = pd.DataFrame({s: fn(prices[s], s) for s in WATCHLIST}).sort_index()
    except Exception as e:
        print(f"{fid}: ERROR {e}")
        continue
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

out = Path('scripts/miner_3_20320624_results_batch36.json')
out.write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
print(f"\nresults saved to {out} ({time.time()-t0:.1f}s total)")
