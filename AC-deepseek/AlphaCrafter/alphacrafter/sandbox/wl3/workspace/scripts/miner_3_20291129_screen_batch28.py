"""miner_3 2029-11-29 novel factor screen (batch 28).

Fresh interpretable candidates NOT in the effective library and NOT previously
evicted/quarantined/rejected (checked against factors/evicted, quarantine,
rejected and prior round results). Vectorized IC engine. Admission gates
(warm-up 2020-01-01..2026-07-15): |IC10|>=0.007, |ICIR10|>=0.084. Drift window
2026-07-16..2029-11-28 on an extended grid. Library corr: max abs mean daily
cross-sectional Spearman vs all top-level *_signal.npy artifacts.
"""
import sys
import json
import time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import rankdata

sys.path.insert(0, 'scripts')
from factor_common import (load_prices, load_index, canonical_grid, WATCHLIST,
                           VAL_START, VAL_END, factor_to_panel, forward_returns)

t0 = time.time()
prices = load_prices(days=2600)
print(f"assets loaded: {len(prices)}; max date: {max(d.index.max() for d in prices.values())} ({time.time()-t0:.1f}s)")

r_all = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()}).sort_index()
cnt = r_all.notna().sum(axis=1)
r_ew = r_all.mean(axis=1)
r_ew[cnt < 8] = np.nan

# extended grid (2020-01-01 .. current) for warm + drift IC
idx = set()
for s, df in prices.items():
    idx.update(df.index)
ext_grid = pd.DatetimeIndex(sorted(idx))
ext_grid = ext_grid[ext_grid >= VAL_START]
print(f"ext grid: {len(ext_grid)} dates, {ext_grid.min().date()}..{ext_grid.max().date()}")
# canonical warm grid
grid = canonical_grid(prices)
print(f"canonical grid: {len(grid)} dates, {grid.min().date()}..{grid.max().date()}")

# observation-only signals (for macro-conditional candidates)
vix = load_index('VIX', prices=prices)
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


candidates = {}

# 1. serial autocorrelation of daily returns over 20d (trend persistence vs whipsaw)
def f_autocorr_20(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), r.shift(1).rename('rl')], axis=1).dropna()
    mu = z['r'].rolling(20).mean()
    num = ((z['r'] - mu) * (z['rl'] - mu)).rolling(20).mean()
    den = (z['r'] - mu).rolling(20).std() * (z['rl'] - mu).rolling(20).std()
    return (num / den).reindex(df.index)
candidates['autocorr_20'] = f_autocorr_20

# 2. beta of asset returns to US10Y yield changes over 60d (yield sensitivity)
def f_us10y_beta_60(df, s):
    if us10y is None:
        return None
    y = us10y['close'].pct_change()
    return rolling_beta_series(df, y, 60)
candidates['us10y_beta_60'] = f_us10y_beta_60

# 3. kurtosis term structure: excess kurtosis 20d minus 60d (tail acceleration)
def f_kurt_term_20_60(df, s):
    r = df['close'].pct_change()
    def ek(w):
        mu = w.mean(); sd = w.std(ddof=0)
        if sd == 0 or not np.isfinite(sd):
            return np.nan
        return float(((w - mu) ** 4).mean() / sd ** 4) - 3.0
    k20 = r.rolling(20, min_periods=15).apply(ek, raw=True)
    k60 = r.rolling(60, min_periods=40).apply(ek, raw=True)
    return (k20 - k60).reindex(df.index)
candidates['kurt_term_20_60'] = f_kurt_term_20_60

# 4. drawdown position: current drawdown depth / max drawdown depth over 60d
def f_dd_pos_60(df, s):
    c = df['close']
    roll_max = c.rolling(60, min_periods=20).max()
    dd = c / roll_max - 1.0
    dd60 = dd.rolling(60, min_periods=20).min()
    return (dd / dd60.replace(0, np.nan)).reindex(df.index)
candidates['dd_pos_60'] = f_dd_pos_60

# 5. trend share: |20d return| / |60d return| (recent move dominance)
def f_trend_share_20_60(df, s):
    c = df['close']
    m20 = c / c.shift(20) - 1.0
    m60 = c / c.shift(60) - 1.0
    return (m20.abs() / m60.abs().replace(0, np.nan)).reindex(df.index)
candidates['trend_share_20_60'] = f_trend_share_20_60

# 6. risk-link diff: corr with SPX minus corr with XAU over 60d (risk-on vs haven linkage)
spx_r = r_all['SPX'] if 'SPX' in r_all else None
xau_r = r_all['XAU'] if 'XAU' in r_all else None
def f_risklink_diff_60(df, s):
    if spx_r is None or xau_r is None:
        return None
    c_spx = rolling_corr_series(df, spx_r, 60)
    c_xau = rolling_corr_series(df, xau_r, 60)
    return (c_spx - c_xau).reindex(df.index)
candidates['risklink_diff_60'] = f_risklink_diff_60

# 7. overnight share of total 20d move: mean |gap| / (mean |gap| + mean intraday |move|)
def f_overnight_share_20(df, s):
    o = df['open']; pc = df['close'].shift(1); c = df['close']
    gap = (o / pc - 1.0).abs()
    intr = (c / o - 1.0).abs()
    g = gap.rolling(20).mean()
    i = intr.rolling(20).mean()
    return (g / (g + i).replace(0, np.nan)).reindex(df.index)
candidates['overnight_share_20'] = f_overnight_share_20

# 8. breakout intensity: count of 20d high/low breaks in last 20d
def f_breakout_count_20(df, s):
    c = df['close']
    hh = c.rolling(20, min_periods=5).max().shift(1)
    ll = c.rolling(20, min_periods=5).min().shift(1)
    up = (c > hh).astype(float)
    dn = (c < ll).astype(float)
    return (up.rolling(20).sum() - dn.rolling(20).sum()).reindex(df.index)
candidates['breakout_count_20'] = f_breakout_count_20

# 9. CS rank persistence: corr of cross-sectional rank with rank 10d ago over 60d
def f_cs_rank_persist_60(df, s):
    r = df['close'].pct_change()
    mom10 = df['close'] / df['close'].shift(10) - 1.0
    # cross-sectional rank of 10d momentum on each date
    ranks = r_all.rolling(10).apply(lambda w: np.nanmean(w), raw=True).rank(axis=1)
    own = ranks[s].reindex(r.index)
    own_lag = own.shift(10)
    z = pd.concat([own.rename('o'), own_lag.rename('ol')], axis=1).dropna()
    mu = z['o'].rolling(60).mean()
    num = ((z['o'] - mu) * (z['ol'] - mu)).rolling(60).mean()
    den = (z['o'] - mu).rolling(60).std() * (z['ol'] - mu).rolling(60).std()
    return (num / den).reindex(df.index)
candidates['cs_rank_persist_60'] = f_cs_rank_persist_60

# 10. VIX-gated vol: 20d vol interacted with VIX level z (vol x stress)
def f_vix_gated_vol_20(df, s):
    if vix is None:
        return None
    v = df['close'].pct_change().rolling(20).std()
    vixz = (vix['close'] - vix['close'].rolling(250).mean()) / vix['close'].rolling(250).std().replace(0, np.nan)
    vixz = vixz.reindex(v.index).ffill()
    return (v * vixz).reindex(df.index)
candidates['vix_gated_vol_20'] = f_vix_gated_vol_20

# ---------- re-validation of strong batch-27 near-misses ----------
# gap_autocorr_60 passed the warm gate in batch27 (ic=0.0252, icir=0.0860) with
# positive recent IC; vol_trend_20_60 failed warm but had very strong recent IC.
def f_gap_autocorr_60(df, s):
    o = df['open']; pc = df['close'].shift(1)
    g = o / pc - 1.0
    z = pd.concat([g.rename('g'), g.shift(1).rename('gl')], axis=1).dropna()
    mu = z['g'].rolling(60).mean()
    num = ((z['g'] - mu) * (z['gl'] - mu)).rolling(60).mean()
    den = (z['g'] - mu).rolling(60).std() * (z['gl'] - mu).rolling(60).std()
    return (num / den).reindex(df.index)
candidates['gap_autocorr_60'] = f_gap_autocorr_60

def f_vol_trend_20_60(df, s):
    v = df['volume']
    if v.notna().sum() < 120:
        return None
    return (v.rolling(20).mean() / v.rolling(60).mean()).reindex(df.index)
candidates['vol_trend_20_60'] = f_vol_trend_20_60

# ---------- IC engine ----------
gidx = ext_grid


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

# library artifacts (canonical grid)
lib_artifacts = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    try:
        arr = np.load(p, allow_pickle=False)
        if arr.shape[0] == len(grid) and arr.shape[1] == 15:
            lib_artifacts[p.name.replace('_signal.npy', '')] = arr
    except Exception:
        pass
print(f"library artifacts for corr audit: {len(lib_artifacts)}")


def max_lib_corr(mat_canon):
    best, best_id = 0.0, None
    n = len(grid)
    for fid, la in lib_artifacts.items():
        corrs = np.full(n, np.nan)
        for i in range(n):
            x = mat_canon[i]; y = la[i]
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


# ---------- run validation ----------
warm = (gidx >= VAL_START) & (gidx <= VAL_END)
rstart = VAL_END + pd.Timedelta(days=1)
recent = gidx >= rstart
# limit recent to dates with enough forward data (horizon 10)
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
    # canonical-grid matrix for library corr
    mat_canon = panel.reindex(grid)[WATCHLIST].values.astype(float)
    rho, fid_rho = max_lib_corr(mat_canon)
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
    print(f"{fid:24s} IC={r['ic']:.4f} ICIR={r['icir']:.4f} rho={r['rho']:.3f} recentIC={r['ic_recent']:.4f} recentICIR={r['icir_recent']:.4f} PASS={'Y' if abs(r['ic'])>=0.007 and abs(r['icir'])>=0.084 else 'N'}")
json.dump(results, open('scripts/miner_3_20291129_results_batch28.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_3_20291129_results_batch28.json; total time %.1fs" % (time.time()-t0))
