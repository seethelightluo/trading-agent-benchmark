"""miner_3 2029-08-09 novel factor screen (batch 27).

Fresh interpretable candidates NOT in the effective library and NOT previously
evicted (checked against factors/evicted and prior round results). Vectorized IC
engine. Admission gates (warm-up 2020-01-01..2026-07-15): |IC10|>=0.007,
|ICIR10|>=0.084. Drift window 2026-07-16..2029-08-08 on an extended grid.
Library corr: max abs mean daily cross-sectional Spearman vs all *_signal.npy.
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

# 1. downside semideviation / upside semideviation over 20d (vol asymmetry)
def f_semi_vol_ratio_20(df, s):
    r = df['close'].pct_change()
    up = r.clip(lower=0.0); dn = (-r).clip(lower=0.0)
    su = up.pow(2).rolling(20).mean().pow(0.5)
    sd = dn.pow(2).rolling(20).mean().pow(0.5)
    return (sd / su.replace(0, np.nan)).reindex(df.index)
candidates['semi_vol_ratio_20'] = f_semi_vol_ratio_20

# 2. excess kurtosis of daily returns over 20d (tail weight)
def f_kurt_20(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(20).mean()
    sd = r.rolling(20).std()
    m4 = ((r - mu) ** 4).rolling(20).mean()
    return ((m4 / sd.pow(4)) - 3.0).reindex(df.index)
candidates['kurt_20'] = f_kurt_20

# 3. max drawdown depth/duration velocity over 60d
def f_dd_velocity_60(df, s):
    c = df['close']
    roll_max = c.rolling(60, min_periods=20).max()
    dd = c / roll_max - 1.0
    depth = dd.rolling(20).min()  # deepest drawdown in last 20d
    # duration: count of days since last 60d high (approx via cummax age)
    age = pd.Series(np.nan, index=c.index)
    cm = c.cummax()
    # days since the running max was set
    pos = np.arange(len(c))
    lastmax_idx = pd.Series(pos, index=c.index).where(c == cm).ffill()
    age = pos - lastmax_idx
    age = pd.Series(age, index=c.index)
    return (depth / (age + 1).replace(0, np.nan)).reindex(df.index)
candidates['dd_velocity_60'] = f_dd_velocity_60

# 4. average pairwise correlation with the other 14 assets over 60d (diversifier score)
r_all_f = r_all
def f_avg_pair_corr_60(df, s):
    r = df['close'].pct_change()
    others = [o for o in WATCHLIST if o != s and o in r_all_f.columns]
    oc = r_all_f[others]
    cs = []
    for o in others:
        z = pd.concat([r.rename('r'), oc[o].rename('o')], axis=1).dropna()
        cs.append(z['r'].rolling(60).corr(z['o']))
    if not cs:
        return None
    mat = pd.concat(cs, axis=1)
    return mat.mean(axis=1).reindex(df.index)
candidates['avg_pair_corr_60'] = f_avg_pair_corr_60

# 5. 20d-60d correlation change vs EW basket (market-correlation regime shift)
def f_corr_ew_change_20_60(df, s):
    r = df['close'].pct_change()
    z = pd.concat([r.rename('r'), r_ew.rename('e')], axis=1).dropna()
    c20 = z['r'].rolling(20).corr(z['e'])
    c60 = z['r'].rolling(60).corr(z['e'])
    return (c20 - c60).reindex(df.index)
candidates['corr_ew_change_20_60'] = f_corr_ew_change_20_60

# 6. beta change (60d beta to EW basket minus 120d beta) - regime shift in exposure
def f_beta_change_60_120(df, s):
    b60 = rolling_beta_series(df, r_ew, 60)
    b120 = rolling_beta_series(df, r_ew, 120)
    return (b60 - b120).reindex(df.index)
candidates['beta_change_60_120'] = f_beta_change_60_120

# 7. overnight gap autocorrelation over 60d (gap persistence)
def f_gap_autocorr_60(df, s):
    o = df['open']; pc = df['close'].shift(1)
    g = o / pc - 1.0
    z = pd.concat([g.rename('g'), g.shift(1).rename('gl')], axis=1).dropna()
    mu = z['g'].rolling(60).mean()
    num = ((z['g'] - mu) * (z['gl'] - mu)).rolling(60).mean()
    den = (z['g'] - mu).rolling(60).std() * (z['gl'] - mu).rolling(60).std()
    return (num / den).reindex(df.index)
candidates['gap_autocorr_60'] = f_gap_autocorr_60

# 8. overnight/intraday volatility share: mean|gap| / mean intraday range
def f_gap_range_share_20(df, s):
    o = df['open']; pc = df['close'].shift(1); c = df['close']; h = df['high']; lo = df['low']
    gap = (o / pc - 1.0).abs()
    intr = (h - lo) / o
    return (gap.rolling(20).mean() / intr.rolling(20).mean().replace(0, np.nan)).reindex(df.index)
candidates['gap_range_share_20'] = f_gap_range_share_20

# 9. volume trend ratio 20d/60d (volume momentum)
def f_vol_trend_20_60(df, s):
    v = df['volume']
    if v.notna().sum() < 120:
        return None
    return (v.rolling(20).mean() / v.rolling(60).mean()).reindex(df.index)
candidates['vol_trend_20_60'] = f_vol_trend_20_60

# 10. low-vol anomaly: 60d vol / cross-sectional median 60d vol (log)
r_all2 = r_all
def f_cs_vol_rank_60(df, s):
    v = df['close'].pct_change().rolling(60).std()
    med = r_all2.rolling(60).std().median(axis=1)
    z = pd.concat([v.rename('v'), med.rename('m')], axis=1).dropna()
    return (np.log(z['v'] / z['m'])).reindex(df.index)
candidates['cs_vol_rank_60'] = f_cs_vol_rank_60

# 11. short-term reversal scaled: -5d return / 20d vol
def f_rev_5_20(df, s):
    r = df['close'].pct_change()
    mom5 = df['close'] / df['close'].shift(5) - 1.0
    v20 = r.rolling(20).std()
    return (-mom5 / v20.replace(0, np.nan)).reindex(df.index)
candidates['rev_5_vol20'] = f_rev_5_20

# 12. trend R2 over 60d (trend quality)
def f_trend_r2_60(df, s):
    c = df['close']
    x = np.arange(60, dtype=float)
    def r2(win):
        y = win.values
        if len(y) < 60 or np.any(~np.isfinite(y)):
            return np.nan
        xm = x.mean(); ym = y.mean()
        num = ((x - xm) * (y - ym)).sum()
        den = np.sqrt(((x - xm) ** 2).sum() * ((y - ym) ** 2).sum())
        if den == 0:
            return np.nan
        r = num / den
        return r * r
    return c.rolling(60, min_periods=60).apply(r2, raw=False).reindex(df.index)
candidates['trend_r2_60'] = f_trend_r2_60

# 13. z-score of 60d vol vs its 250d history (vol regime z)
def f_vol_regime_z_60x250(df, s):
    v = df['close'].pct_change().rolling(60).std()
    mu = v.rolling(250).mean()
    sd = v.rolling(250).std()
    return ((v - mu) / sd.replace(0, np.nan)).reindex(df.index)
candidates['vol_regime_z_60x250'] = f_vol_regime_z_60x250

# 14. cross-sectional z of 20d return (relative momentum)
def f_cs_mom_z_20(df, s):
    mom20 = df['close'] / df['close'].shift(20) - 1.0
    cs_mu = r_all.rolling(20).apply(lambda w: np.nanmean(w), raw=True)
    cs_sd = r_all.rolling(20).apply(lambda w: np.nanstd(w), raw=True)
    z = pd.concat([mom20.rename('m'), cs_mu.rename('mu'), cs_sd.rename('sd')], axis=1)
    return ((z['m'] - z['mu']) / z['sd'].replace(0, np.nan)).reindex(df.index)
candidates['cs_mom_z_20'] = f_cs_mom_z_20

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
json.dump(results, open('scripts/miner_3_20290809_results_batch27.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_3_20290809_results_batch27.json; total time %.1fs" % (time.time()-t0))
