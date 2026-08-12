"""miner_3 2030-04-18 batch-29 screen (VECTORIZED re-run of 2030-02-07 screen).

Same 20 candidates, same admission gates (warm-up 2020-01-01..2026-07-15):
|IC10| >= 0.007, |ICIR10| >= 0.084.

Library-corr audit v2: reconstruct each persisted factor's own signal grid from
its JSON metadata (start/end/n_dates) and compare the candidate on that exact
grid, so artifacts stored on non-2388 grids are also audited.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, WATCHLIST, VAL_START,
                           VAL_END, factor_to_panel, forward_returns)

t0 = time.time()
prices = load_prices(days=2600)
print(f"assets loaded: {len(prices)}; max date: {max(d.index.max() for d in prices.values())} ({time.time()-t0:.1f}s)")

idx = set()
for s, df in prices.items():
    idx.update(df.index)
gidx = pd.DatetimeIndex(sorted(idx))
gidx = gidx[gidx >= VAL_START]
print(f"trading grid: {len(gidx)} dates, {gidx.min().date()}..{gidx.max().date()}")

cal_grid = pd.date_range(VAL_START, VAL_END, freq='D')
print(f"library calendar grid: {len(cal_grid)} dates, {cal_grid.min().date()}..{cal_grid.max().date()}")

r_all = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()}).sort_index()
cnt = r_all.notna().sum(axis=1)
r_ew = r_all.mean(axis=1)
r_ew[cnt < 8] = np.nan

vix = load_index('VIX', prices=prices)
dxy = load_index('DXY', prices=prices)
us10y = prices.get('US10Y')
cn10y = prices.get('CN10Y')


def rolling_beta_series(df, mkt, window):
    r = df['close'].pct_change()
    mm = mkt.reindex(r.index).ffill()
    z = pd.concat([r.rename('r'), mm.rename('m')], axis=1).dropna()
    cov = z['r'].rolling(window).cov(z['m'])
    var = z['m'].rolling(window).var()
    return (cov / var).reindex(df.index)


def rolling_corr_series(df, other, window):
    r = df['close'].pct_change()
    o = other.reindex(r.index).ffill()
    z = pd.concat([r.rename('r'), o.rename('o')], axis=1).dropna()
    c = z['r'].rolling(window).corr(z['o'])
    return c.reindex(df.index)


def rolling_std_series(df, window):
    return df['close'].pct_change().rolling(window).std().reindex(df.index)


wti = prices.get('WTI')
xau = prices.get('XAU')
btc = prices.get('BTC')
eth = prices.get('ETH')

candidates = {}

# ---- fresh batch-29 ideas ----
def f_wti_xau_beta_60(df, s):
    if wti is None or xau is None:
        return None
    spread_r = wti['close'].pct_change() - xau['close'].pct_change()
    return rolling_beta_series(df, spread_r, 60)
candidates['wti_xau_beta_60'] = f_wti_xau_beta_60


def f_btc_divergence_20(df, s):
    if btc is None:
        return None
    return (df['close'] / df['close'].shift(20) - 1.0 -
            (btc['close'] / btc['close'].shift(20) - 1.0)).reindex(df.index)
candidates['btc_divergence_20'] = f_btc_divergence_20


def f_yc_spread_beta_60(df, s):
    if us10y is None or cn10y is None:
        return None
    sp = us10y['close'] - cn10y['close']
    sp_r = sp.pct_change()
    return rolling_beta_series(df, sp_r, 60)
candidates['yc_spread_beta_60'] = f_yc_spread_beta_60


def f_vol_asym_20(df, s):
    r = df['close'].pct_change()
    neg = r.where(r < 0).abs().rolling(20).mean()
    pos = r.where(r > 0).rolling(20).mean()
    return (neg / pos.replace(0, np.nan)).reindex(df.index)
candidates['vol_asym_20'] = f_vol_asym_20


def f_range_mid_pos_20(df, s):
    c = df['close']
    hi = c.rolling(20).max(); lo = c.rolling(20).min()
    mid = (hi + lo) / 2.0
    return ((c - mid) / (hi - lo).replace(0, np.nan)).reindex(df.index)
candidates['range_mid_pos_20'] = f_range_mid_pos_20


def f_gap_ma_10(df, s):
    o = df['open']; pc = df['close'].shift(1)
    g = o / pc - 1.0
    return g.rolling(10).mean().reindex(df.index)
candidates['gap_ma_10'] = f_gap_ma_10


def f_amihud_trend_60(df, s):
    r = df['close'].pct_change().abs()
    dollar = df['close'] * df['volume']
    am = (r / dollar.replace(0, np.nan))
    a60 = am.rolling(60).mean()
    a20 = am.rolling(20).mean()
    return (a60 / a20.replace(0, np.nan)).reindex(df.index)
candidates['amihud_trend_60'] = f_amihud_trend_60


def f_sys_vol_ratio_60(df, s):
    c = rolling_corr_series(df, r_ew, 60).abs()
    v = rolling_std_series(df, 60)
    return (c / v.replace(0, np.nan)).reindex(df.index)
candidates['sys_vol_ratio_60'] = f_sys_vol_ratio_60


def f_hl_symmetry_20(df, s):
    h = df['high']; l = df['low']; c = df['close']
    up = (h - c).abs(); dn = (c - l).abs()
    return (up / dn.replace(0, np.nan)).rolling(20).mean().reindex(df.index)
candidates['hl_symmetry_20'] = f_hl_symmetry_20


def f_crypto_corr_60(df, s):
    if btc is None:
        return None
    return rolling_corr_series(df, btc['close'].pct_change(), 60)
candidates['crypto_corr_60'] = f_crypto_corr_60


def f_dxy_beta_60(df, s):
    if dxy is None:
        return None
    return rolling_beta_series(df, dxy['close'].pct_change(), 60)
candidates['dxy_beta_60'] = f_dxy_beta_60


def f_vol_surge_persist_5_60(df, s):
    v = df['volume']
    if v.notna().sum() < 120:
        return None
    vz = (v - v.rolling(250).mean()) / v.rolling(250).std().replace(0, np.nan)
    return (vz.rolling(5).mean() - vz.rolling(60).mean()).reindex(df.index)
candidates['vol_surge_persist_5_60'] = f_vol_surge_persist_5_60


def f_dd_speed_60(df, s):
    c = df['close']
    roll_max = c.rolling(60, min_periods=20).max()
    dd = c / roll_max - 1.0
    is_high = (c >= roll_max).astype(float)
    grp = is_high.cumsum()
    days_since = grp.groupby(grp).cumcount()
    days_since = days_since.replace(0, np.nan)
    return (dd / days_since).reindex(df.index)
candidates['dd_speed_60'] = f_dd_speed_60


def f_xau_divergence_20(df, s):
    if xau is None:
        return None
    return (df['close'] / df['close'].shift(20) - 1.0 -
            (xau['close'] / xau['close'].shift(20) - 1.0)).reindex(df.index)
candidates['xau_divergence_20'] = f_xau_divergence_20

# ---- re-validation of batch-27/28 gate passers ----
def f_semi_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    total = r.rolling(20).std()
    dn = r.where(r < 0).rolling(20).std()
    return (dn / total.replace(0, np.nan)).reindex(df.index)
candidates['semi_vol_ratio_20'] = f_semi_vol_ratio_20


def f_dd_velocity_60(df, s):
    c = df['close']
    roll_max = c.rolling(60, min_periods=20).max()
    dd = c / roll_max - 1.0
    return (dd - dd.shift(5)).reindex(df.index)
candidates['dd_velocity_60'] = f_dd_velocity_60


def f_avg_pair_corr_60(df, s):
    c = pd.Series(np.nan, index=df.index)
    n_used = 0
    for other in WATCHLIST:
        if other == s or other not in prices:
            continue
        co = rolling_corr_series(df, prices[other]['close'].pct_change(), 60)
        c = c.add(co, fill_value=0)
        n_used += 1
    return (c / n_used).reindex(df.index)
candidates['avg_pair_corr_60'] = f_avg_pair_corr_60


def f_gap_autocorr_60(df, s):
    o = df['open']; pc = df['close'].shift(1)
    g = o / pc - 1.0
    z = pd.concat([g.rename('g'), g.shift(1).rename('gl')], axis=1).dropna()
    mu = z['g'].rolling(60).mean()
    num = ((z['g'] - mu) * (z['gl'] - mu)).rolling(60).mean()
    den = (z['g'] - mu).rolling(60).std() * (z['gl'] - mu).rolling(60).std()
    return (num / den).reindex(df.index)
candidates['gap_autocorr_60'] = f_gap_autocorr_60


spx_r = r_all['SPX'] if 'SPX' in r_all else None
xau_r = r_all['XAU'] if 'XAU' in r_all else None
def f_risklink_diff_60(df, s):
    if spx_r is None or xau_r is None:
        return None
    return (rolling_corr_series(df, spx_r, 60) - rolling_corr_series(df, xau_r, 60)).reindex(df.index)
candidates['risklink_diff_60'] = f_risklink_diff_60


def f_breakout_count_20(df, s):
    c = df['close']
    hh = c.rolling(20, min_periods=5).max().shift(1)
    ll = c.rolling(20, min_periods=5).min().shift(1)
    up = (c > hh).astype(float)
    dn = (c < ll).astype(float)
    return (up.rolling(20).sum() - dn.rolling(20).sum()).reindex(df.index)
candidates['breakout_count_20'] = f_breakout_count_20

# ---------- vectorized IC engine ----------

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


fwd_mats = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd = forward_returns(prices, h).reindex(gidx)
    fwd_mats[h] = fwd[WATCHLIST].values.astype(float)

# ---- library artifacts with per-factor grids from JSON metadata ----
lib_artifacts = {}  # fid -> (grid, matrix)
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
            # fall back to the canonical 2388 calendar grid if shapes match
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
for fid, fn in candidates.items():
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
    icr = ics[10][recent]
    icr = icr[np.isfinite(icr)]
    ic_rmean = float(icr.mean()) if len(icr) >= 30 else float('nan')
    ic_rsd = float(icr.std(ddof=1)) if len(icr) >= 30 else float('nan')
    ic_ricir = ic_rmean / ic_rsd if len(icr) >= 30 and ic_rsd > 0 else float('nan')
    rho, fid_rho = max_lib_corr(panel)
    ok = abs(ic) >= 0.007 and abs(icir) >= 0.084
    results[fid] = dict(ic=ic, icir=icir, hit=hit, cov=cov, ge8=ge8, turn=turn,
                        decay=decay, rho=rho, rho_id=fid_rho,
                        ic_recent=ic_rmean, icir_recent=ic_ricir, n_recent=len(icr),
                        n_warm=len(ic10w))
    print(f"\n{fid}: warm IC={ic:.4f} ICIR={icir:.4f} hit={hit:.3f} cov={cov:.3f} ge8={ge8:.3f} turn={turn:.2f} ({time.time()-t1:.1f}s)")
    print("   decay: " + " ".join(f"{h}:{decay[str(h)]:.4f}" for h in (1, 2, 3, 5, 10, 20)))
    print(f"   recent(2026-07-16+): IC={ic_rmean:.4f} ICIR={ic_ricir:.4f} n={len(icr)}")
    print(f"   max|lib rho|={rho:.4f} vs {fid_rho}")
    print(f"   ADMISSION: {'PASS' if ok else 'FAIL'}")

print("\n=== SUMMARY ===")
for fid, r in results.items():
    print(f"{fid:26s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} rho={r['rho']:.3f} recentIC={r['ic_recent']:+.4f} recentICIR={r['icir_recent']:+.4f} PASS={'Y' if abs(r['ic'])>=0.007 and abs(r['icir'])>=0.084 else 'N'}")
json.dump(results, open('scripts/miner_3_20300418_results_batch29_fast.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_3_20300418_results_batch29_fast.json; total time %.1fs" % (time.time()-t0))
