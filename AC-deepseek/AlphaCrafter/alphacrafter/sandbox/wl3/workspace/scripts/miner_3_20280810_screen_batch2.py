"""miner_3 2028-08-10 batch-2 novel factor screen (cross-asset, interpretable).

Vectorized IC engine for speed. Validation window: 2020-01-01..2026-07-15.
Admission: |IC10| >= 0.007 and |ICIR10| >= 0.084. Drift: recent IC 2026-07-16+.
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
                           VAL_START, VAL_END, factor_to_panel, forward_returns,
                           signal_matrix)

t0 = time.time()
prices = load_prices(days=2200)
print(f"assets loaded: {len(prices)}; max date: {max(d.index.max() for d in prices.values())} ({time.time()-t0:.1f}s)")

r_all = pd.DataFrame({s: d['close'].pct_change() for s, d in prices.items()}).sort_index()
cnt = r_all.notna().sum(axis=1)
r_ew = r_all.mean(axis=1)
r_ew[cnt < 8] = np.nan

usdjpy = load_index('USDJPY', prices=prices)
print("USDJPY loaded:", usdjpy is not None)


def rolling_beta_series(df, mkt, window):
    r = df['close'].pct_change()
    mm = mkt.reindex(r.index).ffill()
    z = pd.concat([r.rename('r'), mm.rename('m')], axis=1).dropna()
    cov = z['r'].rolling(window).cov(z['m'])
    var = z['m'].rolling(window).var()
    return (cov / var).reindex(df.index)


candidates = {}

def f_gap_od_ratio(df, s):
    o = df['open']; c = df['close']; pc = df['close'].shift(1)
    ov = o / pc - 1.0
    intr = c / o - 1.0
    return (ov / intr.replace(0, np.nan)).rolling(20).mean()
candidates['gap_overnight_ratio_20'] = f_gap_od_ratio

def f_upper_shadow(df, s):
    hi = df['high']; lo = df['low']; o = df['open']; c = df['close']
    rng = (hi - lo).replace(0, np.nan)
    return ((hi - np.maximum(o, c)) / rng).rolling(20).mean()
candidates['upper_shadow_20'] = f_upper_shadow

def f_lower_shadow(df, s):
    hi = df['high']; lo = df['low']; o = df['open']; c = df['close']
    rng = (hi - lo).replace(0, np.nan)
    return ((np.minimum(o, c) - lo) / rng).rolling(20).mean()
candidates['lower_shadow_20'] = f_lower_shadow

def f_autocorr5(df, s):
    r = df['close'].pct_change()
    mu = r.rolling(60).mean()
    num = ((r - mu) * (r.shift(5) - mu)).rolling(60).mean()
    den = (r - mu).rolling(60).std() * (r.shift(5) - mu).rolling(60).std()
    return (num / den).reindex(df.index)
candidates['ret_autocorr_5'] = f_autocorr5

def f_bollinger_pos(df, s):
    c = df['close']
    return ((c - c.rolling(20).mean()) / c.rolling(20).std()).reindex(df.index)
candidates['bollinger_pos_20'] = f_bollinger_pos

def f_vol_term_5_60(df, s):
    r = df['close'].pct_change()
    return (r.rolling(5).std() / r.rolling(60).std()).reindex(df.index)
candidates['vol_term_5_60'] = f_vol_term_5_60

def f_dd_depth_60(df, s):
    c = df['close']
    return (c / c.rolling(60, min_periods=20).max() - 1.0).reindex(df.index)
candidates['drawdown_depth_60'] = f_dd_depth_60

r_wti = r_all['WTI']
candidates['wti_beta_60'] = lambda df, s: rolling_beta_series(df, r_wti, 60)

r_xau = r_all['XAU']
candidates['xau_beta_60'] = lambda df, s: rolling_beta_series(df, r_xau, 60)

r_spread = (prices['CN10Y']['close'] - prices['US10Y']['close']).pct_change()
candidates['cnus10y_spread_beta_60'] = lambda df, s: rolling_beta_series(df, r_spread, 60)

def f_er_60(df, s):
    c = df['close']; r = c.pct_change()
    net = (c - c.shift(60)).abs()
    path = r.abs().rolling(60).sum()
    return (net / path).reindex(df.index)
candidates['efficiency_ratio_60'] = f_er_60

def f_upday_share(df, s):
    r = df['close'].pct_change()
    up = r.clip(lower=0.0).rolling(20).mean()
    dn = r.clip(upper=0.0).rolling(20).mean().abs()
    return (up / dn.replace(0, np.nan)).reindex(df.index)
candidates['upday_ret_share_20'] = f_upday_share

def f_range_term(df, s):
    rng = (df['high'] - df['low']) / df['close']
    return (rng.rolling(20).mean() - rng.rolling(60).mean()).reindex(df.index)
candidates['range_term_20_60'] = f_range_term

candidates['mom_250d_skip5'] = lambda df, s: df['close'].shift(5) / df['close'].shift(250) - 1.0

def f_usdjpy_beta(df, s):
    if usdjpy is None:
        return None
    r = df['close'].pct_change()
    v = usdjpy['close'].pct_change()
    j = pd.concat([r.rename('r'), v.rename('v')], axis=1).dropna()
    b = (j['r'].rolling(60).cov(j['v']) / j['v'].rolling(60).var()).reindex(j.index)
    cond = np.sign(usdjpy['close'].shift(1) / usdjpy['close'].shift(21) - 1.0)
    return (-b * cond).reindex(df.index)
candidates['usdjpy_beta_cond_60x20'] = f_usdjpy_beta

# ---------- fast IC engine ----------
grid = canonical_grid(prices)
gidx = pd.Index(grid)


def fast_rank_ic(fmat, rmat, min_valid=8):
    """Row-wise Spearman IC between aligned matrices (n_dates, n_assets)."""
    n = fmat.shape[0]
    ics = np.full(n, np.nan)
    for i in range(n):
        x = fmat[i]; y = rmat[i]
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() >= min_valid:
            rx = rankdata(x[m]); ry = rankdata(y[m])
            ics[i] = np.corrcoef(rx, ry)[0, 1]
    return ics


# forward return matrices on grid
fwd_mats = {}
for h in (1, 2, 3, 5, 10, 20):
    fwd = forward_returns(prices, h).reindex(gidx)
    fwd_mats[h] = fwd[WATCHLIST].values.astype(float)

# library artifacts
lib_artifacts = {}
for p in sorted(Path('factors').glob('*_signal.npy')):
    try:
        arr = np.load(p, allow_pickle=False)
        if arr.shape[0] == len(grid) and arr.shape[1] == 15:
            lib_artifacts[p.name.replace('_signal.npy', '')] = arr
    except Exception:
        pass
print(f"library artifacts for corr audit: {len(lib_artifacts)}")


def max_lib_corr(mat):
    best, best_id = 0.0, None
    n = len(grid)
    for fid, la in lib_artifacts.items():
        corrs = np.full(n, np.nan)
        for i in range(n):
            x = mat[i]; y = la[i]
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
results = {}
for fid, fn in candidates.items():
    panel = factor_to_panel(fn, prices)
    if panel.empty:
        print(f"{fid}: EMPTY panel"); continue
    mat = signal_matrix(panel, grid)
    ics = {}
    for h in (1, 2, 3, 5, 10, 20):
        ics[h] = fast_rank_ic(mat, fwd_mats[h])
    ic10w = ics[10][warm]
    ic10w = ic10w[np.isfinite(ic10w)]
    if len(ic10w) < 100:
        print(f"{fid}: insufficient warm-up IC dates {len(ic10w)}"); continue
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
    rho, fid_rho = max_lib_corr(mat)
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
    print(f"{fid:26s} IC={r['ic']:.4f} ICIR={r['icir']:.4f} rho={r['rho']:.3f} recentIC={r['ic_recent']:.4f} recentICIR={r['icir_recent']:.4f} PASS={'Y' if abs(r['ic'])>=0.007 and abs(r['icir'])>=0.084 else 'N'}")
json.dump(results, open('scripts/miner_3_20280810_results_batch2.json', 'w'), indent=1, default=str)
print("\nsaved scripts/miner_3_20280810_results_batch2.json; total time %.1fs" % (time.time()-t0))
