"""miner_3 2031-07-24: screen batch-33 novel factor candidates on the 15-asset
cross-asset universe. Warm-up admission gates (2020-01-01..2026-07-15):
  |IC10| >= 0.007, |ICIR10| >= 0.084, library |rho| < 0.5 (gate re-checks).

Candidate families (novel vs library): USDJPY/USDCNY conditional macro betas
(library has DXY/EURUSD/VIX only), classic RSI, Bollinger position, return
skewness/kurtosis (library has intraday skew only), max/min daily extreme
asymmetry, return concentration, multi-horizon momentum alignment, candle wick
asymmetry, short/long vol ratio, 120d range position, cross-sectional relative
momentum, gap magnitude (library has gap frequency only).
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
print(f"library calendar grid: {len(cal_grid)} dates, {cal_grid.min().date()}..{cal_grid.max().date()}")

# observation-only macro signals (capped at visible horizon)
usdjpy = load_index('USDJPY', days=3200, prices=prices)
usdcny = load_index('USDCNY', days=3200, prices=prices)
print(f"USDJPY: {None if usdjpy is None else len(usdjpy)} rows {None if usdjpy is None else usdjpy.index.min().date()}..{None if usdjpy is None else usdjpy.index.max().date()}")
print(f"USDCNY: {None if usdcny is None else len(usdcny)} rows {None if usdcny is None else usdcny.index.min().date()}..{None if usdcny is None else usdcny.index.max().date()}")


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


# ---------------- candidate factor functions ----------------
def f_usdjpy_beta_cond_60x20(df, s):
    if usdjpy is None:
        return None
    r = df['close'].pct_change()
    jr = usdjpy['close'].pct_change()
    z = pd.concat([r.rename('r'), jr.rename('j')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['j']) / z['j'].rolling(60, min_periods=30).var()
    trend = usdjpy['close'] / usdjpy['close'].shift(20) - 1.0
    return (b * trend).reindex(df.index)


def f_usdcny_beta_cond_60x20(df, s):
    if usdcny is None:
        return None
    r = df['close'].pct_change()
    cr = usdcny['close'].pct_change()
    z = pd.concat([r.rename('r'), cr.rename('c')], axis=1).dropna()
    b = z['r'].rolling(60, min_periods=30).cov(z['c']) / z['c'].rolling(60, min_periods=30).var()
    trend = usdcny['close'] / usdcny['close'].shift(20) - 1.0
    return (b * trend).reindex(df.index)


def f_rsi_14(df, s):
    c = df['close']
    delta = c.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)
    au = up.rolling(14, min_periods=7).mean()
    ad = dn.rolling(14, min_periods=7).mean()
    rs = au / ad.replace(0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + rs)
    return rsi.reindex(df.index)


def f_bollinger_pos_20(df, s):
    c = df['close']
    ma = c.rolling(20, min_periods=10).mean()
    sd = c.rolling(20, min_periods=10).std()
    return ((c - ma) / (2.0 * sd).replace(0, np.nan)).reindex(df.index)


def f_ret_skew_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=30).skew().reindex(df.index)


def f_ret_kurt_60(df, s):
    r = df['close'].pct_change()
    return r.rolling(60, min_periods=30).kurt().reindex(df.index)


def f_max_min_ratio_20(df, s):
    r = df['close'].pct_change()
    mx = r.rolling(20, min_periods=10).max()
    mn = r.rolling(20, min_periods=10).min()
    return (mx / mn.abs().replace(0, np.nan)).reindex(df.index)


def f_ret_concentration_60(df, s):
    r = df['close'].pct_change().abs()
    def conc(y):
        y = np.asarray(y, dtype=float)
        y = y[np.isfinite(y)]
        if len(y) < 30:
            return np.nan
        srt = np.sort(y)[::-1]
        tot = srt.sum()
        return srt[:3].sum() / tot if tot > 0 else np.nan
    return r.rolling(60, min_periods=30).apply(conc, raw=True).reindex(df.index)


def f_mom_align_3h(df, s):
    c = df['close']
    m10 = c / c.shift(10) - 1.0
    m20 = c / c.shift(20) - 1.0
    m60 = c / c.shift(60) - 1.0
    out = (np.sign(m10) + np.sign(m20) + np.sign(m60)) / 3.0
    return out.reindex(df.index)


def f_wick_ratio_20(df, s):
    h = df['high']; l = df['low']; c = df['close']; o = df['open']
    upw = h - np.maximum(c, o)
    dnw = np.minimum(c, o) - l
    return (upw.rolling(20, min_periods=10).mean() /
            dnw.rolling(20, min_periods=10).mean().replace(0, np.nan)).reindex(df.index)


def f_vol_ratio_5_60(df, s):
    r = df['close'].pct_change()
    v5 = r.rolling(5, min_periods=3).std()
    v60 = r.rolling(60, min_periods=30).std()
    return (v5 / v60.replace(0, np.nan)).reindex(df.index)


def f_range_pos_120(df, s):
    c = df['close']
    mn = c.rolling(120, min_periods=60).min()
    mx = c.rolling(120, min_periods=60).max()
    return ((c - mn) / (mx - mn).replace(0, np.nan)).reindex(df.index)


def f_mom20_raw(df, s):
    c = df['close']
    return (c / c.shift(20) - 1.0).reindex(df.index)


def f_gap_magnitude_20(df, s):
    o = df['open']; c = df['close']
    gap = (o / c.shift(1) - 1.0).abs()
    return gap.rolling(20, min_periods=10).mean().reindex(df.index)


candidates = {
    'usdjpy_beta_cond_60x20': (f_usdjpy_beta_cond_60x20, 'beta(asset_ret, USDJPY_ret, 60) * (USDJPY/USDJPY.shift(20)-1) (yen carry-regime tilt)'),
    'usdcny_beta_cond_60x20': (f_usdcny_beta_cond_60x20, 'beta(asset_ret, USDCNY_ret, 60) * (USDCNY/USDCNY.shift(20)-1) (RMB regime tilt)'),
    'rsi_14': (f_rsi_14, 'classic RSI(14) (overbought/oversold mean reversion)'),
    'bollinger_pos_20': (f_bollinger_pos_20, '(close - SMA20)/(2*STD20) Bollinger position'),
    'ret_skew_60': (f_ret_skew_60, 'skewness of daily returns over 60d (crash-risk asymmetry)'),
    'ret_kurt_60': (f_ret_kurt_60, 'kurtosis of daily returns over 60d (tail heaviness)'),
    'max_min_ratio_20': (f_max_min_ratio_20, 'max daily ret / |min daily ret| over 20d (directional extreme asymmetry)'),
    'ret_concentration_60': (f_ret_concentration_60, 'top-3 |daily ret| share of total |ret| over 60d (return herding)'),
    'mom_align_3h': (f_mom_align_3h, 'mean sign of 10/20/60d momentum (multi-horizon trend alignment)'),
    'wick_ratio_20': (f_wick_ratio_20, 'mean upper wick / mean lower wick over 20d (supply/demand asymmetry)'),
    'vol_ratio_5_60': (f_vol_ratio_5_60, 'STD5(ret)/STD60(ret) (short-term vol spike vs long-term)'),
    'range_pos_120': (f_range_pos_120, '(close - min120)/(max120 - min120) 120d range position'),
    'rel_mom_20': (f_mom20_raw, '20d momentum minus cross-sectional median (relative momentum)'),
    'gap_magnitude_20': (f_gap_magnitude_20, 'mean |open/prev_close - 1| over 20d (gap magnitude)'),
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
    if fid == 'rel_mom_20':
        panel = panel.sub(panel.median(axis=1), axis=0)  # cross-sectional demean
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
    ok = abs(ic) >= 0.007 and abs(icir) >= 0.084
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
    print(f"  ADMISSION: |IC|={abs(ic):.4f}>=0.007 {abs(ic)>=0.007} | |ICIR|={abs(icir):.4f}>=0.084 {abs(icir)>=0.084} -> {'PASS' if ok else 'FAIL'}")

out = Path('scripts/miner_3_20310724_results_batch33.json')
out.write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
print(f"\nSaved results to {out} ({time.time()-t0:.1f}s)")
