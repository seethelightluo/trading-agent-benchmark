"""miner_3 2030-02-07 novel factor screen (batch 29).

Fresh interpretable candidates + re-validation of batch-27/28 gate-passing
near-misses (semi_vol_ratio_20, dd_velocity_60, avg_pair_corr_60,
gap_autocorr_60, risklink_diff_60, breakout_count_20) which were NOT persisted
because the previous library-corr audit returned rho=0.000 (grid mismatch bug:
candidate on trading-date grid vs library artifacts on 2388-calendar-day grid).

This script fixes the audit: library correlation is computed on the SAME
2388-day calendar grid (2020-01-01..2026-07-15) used by all persisted
*_signal.npy artifacts, so the gate's pairwise Spearman rho is meaningful.

Admission gates (warm-up 2020-01-01..2026-07-15): |IC10| >= 0.007, |ICIR10| >= 0.084.
Drift window 2026-07-16..2030-02-06 reported for context.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, WATCHLIST, VAL_START,
                           VAL_END, factor_to_panel, forward_returns)

t0 = time.time()
prices = load_prices(days=2600)
print(f"assets loaded: {len(prices)}; max date: {max(d.index.max() for d in prices.values())} ({time.time()-t0:.1f}s)")

# trading-date grid for IC (union of asset dates)
idx = set()
for s, df in prices.items():
    idx.update(df.index)
gidx = pd.DatetimeIndex(sorted(idx))
gidx = gidx[gidx >= VAL_START]
print(f"trading grid: {len(gidx)} dates, {gidx.min().date()}..{gidx.max().date()}")

# canonical calendar grid used by persisted library artifacts
cal_grid = pd.date_range(VAL_START, VAL_END, freq='D')
print(f"library calendar grid: {len(cal_grid)} dates, {cal_grid.min().date()}..{cal_grid.max().date()}")

r_all = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()}).sort_index()
cnt = r_all.notna().sum(axis=1)
r_ew = r_all.mean(axis=1)
r_ew[cnt < 8] = np.nan

# observation-only signals
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

# 1. WTI-XAU spread beta over 60d (energy-vs-haven relative sensitivity)
def f_wti_xau_beta_60(df, s):
    if wti is None or xau is None:
        return None
    spread_r = wti['close'].pct_change() - xau['close'].pct_change()
    return rolling_beta_series(df, spread_r, 60)
candidates['wti_xau_beta_60'] = f_wti_xau_beta_60

# 2. BTC divergence: asset 20d return minus BTC 20d return (crypto-beta divergence)
def f_btc_divergence_20(df, s):
    if btc is None:
        return None
    return (df['close'] / df['close'].shift(20) - 1.0 -
            (btc['close'] / btc['close'].shift(20) - 1.0)).reindex(df.index)
candidates['btc_divergence_20'] = f_btc_divergence_20

# 3. Yield-curve spread (US10Y-CN10Y) beta over 60d
def f_yc_spread_beta_60(df, s):
    if us10y is None or cn10y is None:
        return None
    sp = us10y['close'] - cn10y['close']
    sp_r = sp.pct_change()
    return rolling_beta_series(df, sp_r, 60)
candidates['yc_spread_beta_60'] = f_yc_spread_beta_60

# 4. Downside asymmetry: mean |neg ret| / mean pos ret over 20d
def f_vol_asym_20(df, s):
    r = df['close'].pct_change()
    neg = r.where(r < 0).abs().rolling(20).mean()
    pos = r.where(r > 0).rolling(20).mean()
    return (neg / pos.replace(0, np.nan)).reindex(df.index)
candidates['vol_asym_20'] = f_vol_asym_20

# 5. Range-mid position: (close - 20d midpoint) / 20d range
def f_range_mid_pos_20(df, s):
    c = df['close']
    hi = c.rolling(20).max(); lo = c.rolling(20).min()
    mid = (hi + lo) / 2.0
    return ((c - mid) / (hi - lo).replace(0, np.nan)).reindex(df.index)
candidates['range_mid_pos_20'] = f_range_mid_pos_20

# 6. Persistent gap direction: 10d mean overnight gap
def f_gap_ma_10(df, s):
    o = df['open']; pc = df['close'].shift(1)
    g = o / pc - 1.0
    return g.rolling(10).mean().reindex(df.index)
candidates['gap_ma_10'] = f_gap_ma_10

# 7. Liquidity trend: Amihud(60d) / Amihud(20d) (rising = deteriorating liquidity)
def f_amihud_trend_60(df, s):
    r = df['close'].pct_change().abs()
    dollar = df['close'] * df['volume']
    am = (r / dollar.replace(0, np.nan))
    a60 = am.rolling(60).mean()
    a20 = am.rolling(20).mean()
    return (a60 / a20.replace(0, np.nan)).reindex(df.index)
candidates['amihud_trend_60'] = f_amihud_trend_60

# 8. Systematic-ness per unit risk: |corr with EW| / own vol over 60d
def f_sys_vol_ratio_60(df, s):
    c = rolling_corr_series(df, r_ew, 60).abs()
    v = rolling_std_series(df, 60)
    return (c / v.replace(0, np.nan)).reindex(df.index)
candidates['sys_vol_ratio_60'] = f_sys_vol_ratio_60

# 9. High-low symmetry: (high-close)/(close-low) averaged over 20d
def f_hl_symmetry_20(df, s):
    h = df['high']; l = df['low']; c = df['close']
    up = (h - c).abs(); dn = (c - l).abs()
    return (up / dn.replace(0, np.nan)).rolling(20).mean().reindex(df.index)
candidates['hl_symmetry_20'] = f_hl_symmetry_20

# 10. Crypto linkage: rolling corr with BTC over 60d
def f_crypto_corr_60(df, s):
    if btc is None:
        return None
    return rolling_corr_series(df, btc['close'].pct_change(), 60)
candidates['crypto_corr_60'] = f_crypto_corr_60

# 11. DXY beta over 60d (USD sensitivity)
def f_dxy_beta_60(df, s):
    if dxy is None:
        return None
    return rolling_beta_series(df, dxy['close'].pct_change(), 60)
candidates['dxy_beta_60'] = f_dxy_beta_60

# 12. Volume surge persistence: mean vol z (5d) minus mean vol z (60d)
def f_vol_surge_persist_5_60(df, s):
    v = df['volume']
    if v.notna().sum() < 120:
        return None
    vz = (v - v.rolling(250).mean()) / v.rolling(250).std().replace(0, np.nan)
    return (vz.rolling(5).mean() - vz.rolling(60).mean()).reindex(df.index)
candidates['vol_surge_persist_5_60'] = f_vol_surge_persist_5_60

# 13. Drawdown speed: current 60d drawdown depth / days since 60d high
def f_dd_speed_60(df, s):
    c = df['close']
    roll_max = c.rolling(60, min_periods=20).max()
    dd = c / roll_max - 1.0
    days_since_high = pd.Series(np.nan, index=c.index)
    rm = roll_max.dropna()
    for d in rm.index:
        w = c.loc[:d].tail(60)
        last_hi = w[w == w.max()].index[-1]
        days_since_high.loc[d] = (d - last_hi).days
    speed = (dd / days_since_high.replace(0, np.nan)).reindex(df.index)
    return speed
candidates['dd_speed_60'] = f_dd_speed_60

# 14. XAU divergence: asset 20d return minus XAU 20d return (haven divergence)
def f_xau_divergence_20(df, s):
    if xau is None:
        return None
    return (df['close'] / df['close'].shift(20) - 1.0 -
            (xau['close'] / xau['close'].shift(20) - 1.0)).reindex(df.index)
candidates['xau_divergence_20'] = f_xau_divergence_20

# ---- re-validation of batch-27/28 gate passers (proper rho audit) ----

# semi_vol_ratio_20: downside semi-vol / total vol over 20d
def f_semi_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    total = r.rolling(20).std()
    dn = r.where(r < 0).rolling(20).std()
    return (dn / total.replace(0, np.nan)).reindex(df.index)
candidates['semi_vol_ratio_20'] = f_semi_vol_ratio_20

# dd_velocity_60: 60d drawdown depth change per day (speed of drawdown)
def f_dd_velocity_60(df, s):
    c = df['close']
    roll_max = c.rolling(60, min_periods=20).max()
    dd = c / roll_max - 1.0
    return (dd - dd.shift(5)).reindex(df.index)
candidates['dd_velocity_60'] = f_dd_velocity_60

# avg_pair_corr_60: average pairwise rolling corr of asset with all others over 60d
def f_avg_pair_corr_60(df, s):
    c = pd.Series(np.nan, index=df.index)
    for other in WATCHLIST:
        if other == s or other not in prices:
            continue
        co = rolling_corr_series(df, prices[other]['close'].pct_change(), 60)
        c = c.add(co, fill_value=0)
    return (c / (len(WATCHLIST) - 1)).reindex(df.index)
candidates['avg_pair_corr_60'] = f_avg_pair_corr_60

# gap_autocorr_60: autocorrelation of overnight gaps over 60d
def f_gap_autocorr_60(df, s):
    o = df['open']; pc = df['close'].shift(1)
    g = o / pc - 1.0
    z = pd.concat([g.rename('g'), g.shift(1).rename('gl')], axis=1).dropna()
    mu = z['g'].rolling(60).mean()
    num = ((z['g'] - mu) * (z['gl'] - mu)).rolling(60).mean()
    den = (z['g'] - mu).rolling(60).std() * (z['gl'] - mu).rolling(60).std()
    return (num / den).reindex(df.index)
candidates['gap_autocorr_60'] = f_gap_autocorr_60

# risklink_diff_60: corr with SPX minus corr with XAU over 60d
spx_r = r_all['SPX'] if 'SPX' in r_all else None
xau_r = r_all['XAU'] if 'XAU' in r_all else None
def f_risklink_diff_60(df, s):
    if spx_r is None or xau_r is None:
        return None
    return (rolling_corr_series(df, spx_r, 60) - rolling_corr_series(df, xau_r, 60)).reindex(df.index)
candidates['risklink_diff_60'] = f_risklink_diff_60

# breakout_count_20: 20d high-break count minus low-break count over 20d
def f_breakout_count_20(df, s):
    c = df['close']
    hh = c.rolling(20, min_periods=5).max().shift(1)
    ll = c.rolling(20, min_periods=5).min().shift(1)
    up = (c > hh).astype(float)
    dn = (c < ll).astype(float)
    return (up.rolling(20).sum() - dn.rolling(20).sum()).reindex(df.index)
candidates['breakout_count_20'] = f_breakout_count_20

# ---------- IC engine ----------

def fast_rank_ic(fmat, rmat, min_valid=8):
    n = fmat.shape[0]
    ics = np.full(n, np.nan)
    for i in range(n):
        x = fmat[i]; y = rmat[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            rx = rankdata(x[m]); ry = rankdata(y[m])
            ics[i] = np.corrcoef(rx, ry)[0, 1]
    return ics


fwd_mats = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd = forward_returns(prices, h).reindex(gidx)
    fwd_mats[h] = fwd[WATCHLIST].values.astype(float)

# library artifacts on the 2388 calendar grid
lib_artifacts = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    try:
        arr = np.load(p, allow_pickle=False)
        if arr.shape[0] == len(cal_grid) and arr.shape[1] == 15:
            lib_artifacts[p.name.replace('_signal.npy', '')] = arr
        else:
            print(f"  skip artifact {p.name} shape {arr.shape}")
    except Exception:
        pass
print(f"library artifacts aligned to calendar grid: {len(lib_artifacts)}")


def max_lib_corr(mat_cal):
    best, best_id = 0.0, None
    n = len(cal_grid)
    for fid, la in lib_artifacts.items():
        corrs = np.full(n, np.nan)
        for i in range(n):
            x = mat_cal[i]; y = la[i]
            m = np.isfinite(x) & np.isfinite(y)
            if m.sum() >= 8:
                rx = rankdata(x[m]); ry = rankdata(y[m])
                corrs[i] = np.corrcoef(rx, ry)[0, 1]
        c = corrs[np.isfinite(corrs)]
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
    panel = factor_to_panel(fn, prices)
    if panel.empty:
        print(f"{fid}: EMPTY panel"); continue
    mat = panel.reindex(gidx)[WATCHLIST].values.astype(float)
    ics = {}
    for h in (1, 2, 3, 5, 10, 20):
        ics[h] = fast_rank_ic(mat, fwd_mats[h])
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
    # library corr on the SAME calendar grid as persisted artifacts
    mat_cal = panel.reindex(cal_grid)[WATCHLIST].values.astype(float)
    rho, fid_rho = max_lib_corr(mat_cal)
    ok = abs(ic) >= 0.007 and abs(icir) >= 0.084
    results[fid] = dict(ic=ic, icir=icir, hit=hit, cov=cov, ge8=ge8, turn=turn,
                        decay=decay, rho=rho, rho_id=fid_rho,
                        ic_recent=ic_rmean, icir_recent=ic_ricir, n_recent=len(icr),
                        n_warm=len(ic10w))
    print(f"\n{fid}: warm IC={ic:.4f} ICIR={icir:.4f} hit={hit:.3f} cov={cov:.3f} ge8={ge8:.3f} turn={turn:.2f}")
    print("   decay: " + " ".join(f"{h}:{decay[str(h)]:.4f}" for h in (1, 2, 3, 5, 10, 20)))
    print(f"   recent(2026-07-16+): IC={ic_rmean:.4f} ICIR={ic_ricir:.4f} n={len(icr)}")
    print(f"   max|lib rho|={rho:.4f} vs {fid_rho}")
    print(f"   ADMISSION: {'PASS' if ok else 'FAIL'}")

print("\n=== SUMMARY ===")
for fid, r in results.items():
    print(f"{fid:26s} IC={r['ic']:+.4f} ICIR={r['icir']:+.4f} rho={r['rho']:.3f} recentIC={r['ic_recent']:+.4f} recentICIR={r['icir_recent']:+.4f} PASS={'Y' if abs(r['ic'])>=0.007 and abs(r['icir'])>=0.084 else 'N'}")
json.dump(results, open('scripts/miner_3_20270207_results_batch29.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_3_20270207_results_batch29.json; total time %.1fs" % (time.time()-t0))
