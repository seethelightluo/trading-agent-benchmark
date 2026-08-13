"""miner2 2034-07-14: fast vectorized re-validation of full factor library on panel through 2034-07-13.
Vectorized daily rank IC (Spearman via cross-sectional rank standardization, NaN-safe).
Gates: abs(IC1) >= 0.0070, abs(ICIR1) >= 0.0840 (same-horizon admission).
Windows: full / 2y / 1y / 6m for drift. Reports max_abs_library_correlation.
"""
import pandas as pd, numpy as np, json, time
sys_path = None
import sys
sys.path.insert(0, 'scripts')

t0 = time.time()
with open('scripts/panel_cache_20340714.pkl', 'rb') as f:
    panel = pd.read_pickle(f)
close = panel['close']; open_ = panel['open']; high = panel['high']; low = panel['low']
ret = panel['ret']; macro = panel['macro']

# ---- signal constructions (must match persisted factor definitions) ----
sig = {}
for nd in [1, 2, 3, 5]:
    sig[f'rev_{nd}d'] = -(np.log(close) - np.log(close.shift(nd)))
sig['rev_1d_vs'] = -(np.log(close) - np.log(close.shift(1))) / ret.rolling(20).std()
sig['id_rev_1d'] = -(close / open_ - 1.0)
sig['nbody_1d'] = -(close - open_) / (high - low)
for nd in [1, 2, 3, 5]:
    hl = high.rolling(nd).max() - low.rolling(nd).min()
    sig[f'nclv_{nd}d'] = -(close - low.rolling(nd).min()) / hl
sig['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
sig['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
vix = macro['VIX']
vix_ret = vix.pct_change()
beta60 = ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
sig['vix_beta_cond_60x20'] = -beta60 * (vix / vix.shift(20) - 1.0)

print(f"signals built in {time.time()-t0:.1f}s", flush=True)

def zscore_rows(X):
    """Cross-sectional z-score per row (NaN-safe). Returns (Z, valid_count)."""
    X = X.astype(float)
    cnt = X.notna().sum(axis=1).to_numpy()
    mu = X.sum(axis=1, min_count=1).to_numpy() / np.maximum(cnt, 1)
    c = X.sub(mu, axis=0)
    ss = (c * c).sum(axis=1, min_count=1).to_numpy()
    sd = np.sqrt(ss / np.maximum(cnt - 1, 1))
    Z = c.div(sd, axis=0)
    return Z, cnt

def vec_daily_rank_ic(sig_df, close, h, min_n=8, start=None, end=None):
    """Vectorized daily Spearman IC between signal and h-day forward return."""
    fwd = close.shift(-h) / close - 1.0
    if start is not None:
        sig_df = sig_df[sig_df.index >= start]
    if end is not None:
        sig_df = sig_df[sig_df.index <= end]
    idx = sig_df.index.intersection(fwd.index)
    if len(idx) == 0:
        return np.array([]), np.array([])
    S = sig_df.loc[idx]
    F = fwd.loc[idx]
    Zs, cnts = zscore_rows(S)
    Zf, _ = zscore_rows(F)
    both = Zs.notna().to_numpy() & Zf.notna().to_numpy()
    prod = (Zs.to_numpy() * Zf.to_numpy())
    prod = np.where(both, prod, np.nan)
    with np.errstate(invalid='ignore'):
        ic = np.nanmean(prod, axis=1)
    valid = (cnts >= min_n) & np.isfinite(ic)
    return ic[valid], idx[valid]

def eval_factor_vec(sig_df, close, horizons=(1, 2, 3, 5, 10), min_n=8, start=None, end=None):
    out = {}
    for h in horizons:
        ics, dates = vec_daily_rank_ic(sig_df, close, h, min_n=min_n, start=start, end=end)
        if len(ics) == 0:
            out[h] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
            continue
        ic = float(np.mean(ics))
        sd = float(np.std(ics, ddof=1))
        icir = ic / sd if sd > 0 else np.nan
        hit = float(np.mean(ics > 0))
        out[h] = dict(ic=ic, icir=icir, hit=hit, n=len(ics))
    s = sig_df[sig_df.index >= start] if start is not None else sig_df
    if end is not None:
        s = s[s.index <= end]
    out['coverage'] = float(s.notna().mean().mean()) if s.shape[0] else 0.0
    rp = s.rank(axis=1, pct=True)
    out['turnover_1d_rank'] = float(rp.diff().abs().mean().mean()) if rp.shape[0] > 1 else 0.0
    out['n_dates'] = int(s.shape[0])
    return out

end = close.index.max()
windows = {
    'full': (None, None),
    'recent_2y': (end - pd.Timedelta(days=730), None),
    'recent_1y': (end - pd.Timedelta(days=365), None),
    'recent_6m': (end - pd.Timedelta(days=183), None),
}

results = {}
for name, s in sig.items():
    results[name] = {}
    for wname, (ws, we) in windows.items():
        results[name][wname] = eval_factor_vec(s, close, horizons=(1, 2, 3, 5, 10), min_n=8, start=ws, end=we)
print(f"evaluation done in {time.time()-t0:.1f}s", flush=True)

# ---- max_abs_library_correlation (pairwise signal rho over common dates, full window) ----
names = list(sig.keys())
n = len(names)
rho_mat = np.eye(n)
for i in range(n):
    for j in range(i + 1, n):
        a = sig[names[i]].stack(); b = sig[names[j]].stack()
        m = a.notna() & b.notna()
        if m.sum() > 30:
            rho_mat[i, j] = rho_mat[j, i] = float(a[m].corr(b[m]))

print("=" * 100)
print(f"FACTOR LIBRARY REVALIDATION - panel through {end.date()}, {close.shape[0]} dates, {close.shape[1]} instruments")
print("=" * 100)
admitted = {}
for name in names:
    print(f"\n### {name}")
    for wname in ['full', 'recent_2y', 'recent_1y', 'recent_6m']:
        r = results[name][wname]
        h1 = r.get(1, {}); h5 = r.get(5, {}); h10 = r.get(10, {})
        print(f"  {wname:10s} IC1={h1.get('ic',np.nan):+.4f} ICIR1={h1.get('icir',np.nan):+.3f} hit1={h1.get('hit',np.nan):.3f} n={h1.get('n',0)} | "
              f"IC5={h5.get('ic',np.nan):+.4f} ICIR5={h5.get('icir',np.nan):+.3f} | IC10={h10.get('ic',np.nan):+.4f} ICIR10={h10.get('icir',np.nan):+.3f} | "
              f"cov={r.get('coverage',np.nan):.3f} turn1d={r.get('turnover_1d_rank',np.nan):.3f}")
    r = results[name]['full']
    h1 = r.get(1, {})
    ic1, icir1 = h1.get('ic', np.nan), h1.get('icir', np.nan)
    if np.isfinite(ic1) and np.isfinite(icir1) and abs(ic1) >= 0.0070 and abs(icir1) >= 0.0840:
        admitted[name] = (ic1, icir1)

print("\n" + "=" * 100)
print("ADMISSION GATE CHECK (full-window 1d: |IC1|>=0.0070 AND |ICIR1|>=0.0840)")
for name, (ic1, icir1) in sorted(admitted.items(), key=lambda kv: -abs(kv[1][0]) * abs(kv[1][1])):
    print(f"  PASS {name:28s} IC1={ic1:+.4f} ICIR1={icir1:+.3f}")
if not admitted:
    print("  (none passed on full window)")

print("\nMax abs pairwise signal correlation (full window, all library factors):")
print("  max_rho =", round(float(np.max(np.abs(rho_mat[np.triu_indices(n, 1)]))), 4))

with open('scripts/miner2_reval_20340714.json', 'w') as f:
    json.dump({k: {wk: {h: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                            for kk, vv in v.items()} for h, v in w.items()}
                   for wk, w in v.items()} for k, v in results.items()}, f, indent=1)
print("\nsaved scripts/miner2_reval_20340714.json")
print(f"total time {time.time()-t0:.1f}s")
