"""miner_3 2031-03-20: screen batch-32 novel factor candidates on the 15-asset
cross-asset universe. Warm-up admission gates (2020-01-01..2026-07-15):
  |IC10| >= 0.007, |ICIR10| >= 0.084, library |rho| < 0.5 (gate re-checks).

Candidate families (novel vs library): trend consistency (R2), downside risk
share, daily close-in-range position, return autocorrelation, volume trend,
overnight/intraday momentum split, candle body ratio, volume-|ret| coupling,
drawdown depth, high-low asymmetry, signed overnight share.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (load_prices, WATCHLIST, VAL_START, VAL_END,
                           factor_to_panel, forward_returns, persist_factor)

t0 = time.time()
prices = load_prices(days=3000)
print(f"assets loaded: {len(prices)}; max date: {max(d.index.max() for d in prices.values())} ({time.time()-t0:.1f}s)")

idx = set()
for s, df in prices.items():
    idx.update(df.index)
gidx = pd.DatetimeIndex(sorted(idx))
gidx = gidx[gidx >= VAL_START]
print(f"trading grid: {len(gidx)} dates, {gidx.min().date()}..{gidx.max().date()}")

cal_grid = pd.date_range(VAL_START, VAL_END, freq='D')
print(f"library calendar grid: {len(cal_grid)} dates, {cal_grid.min().date()}..{cal_grid.max().date()}")


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
def f_r2_trend_60(df, s):
    c = df['close']
    x = np.arange(len(c), dtype=float)
    def r2(y):
        if len(y) < 30 or np.std(y) == 0:
            return np.nan
        b = np.polyfit(x[:len(y)], y, 1)
        pred = np.polyval(b, x[:len(y)])
        ss_res = np.sum((y - pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return c.rolling(60, min_periods=30).apply(r2, raw=True).reindex(df.index)


def f_downside_vol_ratio_60(df, s):
    r = df['close'].pct_change()
    neg = r.where(r < 0, 0.0)
    dstd = neg.rolling(60, min_periods=30).std()
    tstd = r.rolling(60, min_periods=30).std()
    return (dstd / tstd.replace(0, np.nan)).reindex(df.index)


def f_close_pos_range_20(df, s):
    hl = df['high'] - df['low']
    pos = (df['close'] - df['low']) / hl.replace(0, np.nan)
    return pos.rolling(20, min_periods=10).mean().reindex(df.index)


def f_ret_autocorr_60(df, s):
    r = df['close'].pct_change()
    ac = r.rolling(60, min_periods=30).apply(lambda y: pd.Series(y).autocorr(lag=1) if len(y) > 3 else np.nan, raw=False)
    return ac.reindex(df.index)


def f_volume_trend_20_60(df, s):
    v = df['volume']
    return (v.rolling(20, min_periods=10).mean() / v.rolling(60, min_periods=30).mean().replace(0, np.nan)).reindex(df.index)


def f_overnight_mom_20(df, s):
    o = df['open']; c = df['close']
    night = o / c.shift(1) - 1.0
    return night.rolling(20, min_periods=10).sum().reindex(df.index)


def f_intraday_mom_20(df, s):
    o = df['open']; c = df['close']
    intra = c / o - 1.0
    return intra.rolling(20, min_periods=10).sum().reindex(df.index)


def f_body_ratio_20(df, s):
    o = df['open']; h = df['high']; l = df['low']; c = df['close']
    body = (c - o).abs()
    rng = (h - l).replace(0, np.nan)
    return (body / rng).rolling(20, min_periods=10).mean().reindex(df.index)


def f_vol_ret_corr_20(df, s):
    r = df['close'].pct_change().abs()
    v = df['volume']
    c = r.rolling(20, min_periods=10).corr(v)
    return c.reindex(df.index)


def f_drawdown_depth_120(df, s):
    c = df['close']
    rmax = c.rolling(120, min_periods=60).max()
    return (c / rmax - 1.0).reindex(df.index)


def f_hilo_asym_60(df, s):
    c = df['close']
    up = c.rolling(60, min_periods=30).max() - c
    dn = c - c.rolling(60, min_periods=30).min()
    return ((up - dn) / (up + dn).replace(0, np.nan)).reindex(df.index)


def f_overnight_share_signed_20(df, s):
    o = df['open']; c = df['close']
    night = o / c.shift(1) - 1.0
    intra = c / o - 1.0
    ns = night.rolling(20, min_periods=10).sum()
    iss = intra.rolling(20, min_periods=10).sum()
    return ((ns - iss) / (ns.abs() + iss.abs()).replace(0, np.nan)).reindex(df.index)


def f_gap_fill_corr_20(df, s):
    o = df['open']; c = df['close']
    gap = o / c.shift(1) - 1.0
    intra = c / o - 1.0
    g = gap.rolling(20, min_periods=10).corr(intra)
    return g.reindex(df.index)


candidates = {
    'r2_trend_60': (f_r2_trend_60, 'R2 of linear trend fit over 60d (trend consistency)'),
    'downside_vol_ratio_60': (f_downside_vol_ratio_60, 'downside std / total std of daily rets over 60d (risk asymmetry)'),
    'close_pos_range_20': (f_close_pos_range_20, 'mean daily (close-low)/(high-low) over 20d (close-in-range pressure)'),
    'ret_autocorr_60': (f_ret_autocorr_60, 'lag-1 autocorrelation of daily returns over 60d (trend persistence)'),
    'volume_trend_20_60': (f_volume_trend_20_60, '20d avg volume / 60d avg volume (activity expansion)'),
    'overnight_mom_20': (f_overnight_mom_20, 'sum of overnight returns over 20d (overnight momentum)'),
    'intraday_mom_20': (f_intraday_mom_20, 'sum of intraday (open->close) returns over 20d (intraday momentum)'),
    'body_ratio_20': (f_body_ratio_20, 'mean |close-open|/(high-low) over 20d (candle decisiveness)'),
    'vol_ret_corr_20': (f_vol_ret_corr_20, 'corr(|return|, volume) over 20d (volume-activity coupling)'),
    'drawdown_depth_120': (f_drawdown_depth_120, 'close/rolling_max(close,120)-1 (current drawdown depth)'),
    'hilo_asym_60': (f_hilo_asym_60, '(up-dist - down-dist)/(up+down) over 60d (high-low asymmetry)'),
    'overnight_share_signed_20': (f_overnight_share_signed_20, '(night-intra)/(|night|+|intra|) over 20d (signed overnight share)'),
    'gap_fill_corr_20': (f_gap_fill_corr_20, 'corr(gap, intraday return) over 20d (gap continuation vs fill)'),
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

out = Path('scripts/miner_3_20310320_results_batch32.json')
out.write_text(json.dumps(results, indent=2, default=str), encoding='utf-8')
print(f"\nSaved results to {out}")
