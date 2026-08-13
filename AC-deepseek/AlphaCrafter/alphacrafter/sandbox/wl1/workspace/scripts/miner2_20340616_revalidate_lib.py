"""miner2 2034-06-16: re-validate full factor library on fresh panel through 2034-06-15.
Gates: abs(IC1) >= 0.0070, abs(ICIR1) >= 0.0840 (same-horizon admission).
Also reports 2y/1y/6m windows for drift, plus max_abs_library_correlation.
"""
import pandas as pd, numpy as np, json, sys
sys.path.insert(0, 'scripts')
from miner2_val_lib import load_panel, fwd_ret, daily_rank_ic, eval_factor, summarize, WATCH

panel = load_panel('scripts/panel_cache_20340616.pkl')
close = panel['close']; open_ = panel['open']; high = panel['high']; low = panel['low']
ret = panel['ret']; macro = panel['macro']

# ---- signal constructions (must match persisted factor definitions) ----
sig = {}
# miner2 reversal family
for nd in [1, 2, 3, 5]:
    sig[f'rev_{nd}d'] = -(np.log(close) - np.log(close.shift(nd)))
sig['rev_1d_vs'] = -(np.log(close) - np.log(close.shift(1))) / ret.rolling(20).std()
# intraday family
sig['id_rev_1d'] = -(close / open_ - 1.0)
sig['nbody_1d'] = -(close - open_) / (high - low)
for nd in [1, 2, 3, 5]:
    hl = high.rolling(nd).max() - low.rolling(nd).min()
    sig[f'nclv_{nd}d'] = -(close - low.rolling(nd).min()) / hl
# momentum
sig['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
# vol of vol
sig['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
# vix beta conditional
vix = macro['VIX']
vix_ret = vix.pct_change()
beta60 = ret.rolling(60).cov(vix_ret) / vix_ret.rolling(60).var()
sig['vix_beta_cond_60x20'] = -beta60 * (vix / vix.shift(20) - 1.0)

# ---- evaluation windows ----
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
        res = eval_factor(s, close, horizons=(1, 2, 3, 5, 10), min_n=8, start=ws, end=we)
        results[name][wname] = res

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
print(f"FACTOR LIBRARY REVALIDATION — panel through {end.date()}, {close.shape[0]} dates, {close.shape[1]} instruments")
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
    # admission check on full-window 1d (same-horizon as persisted)
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

# persist per-factor results for audit
with open('scripts/miner2_reval_20340616.json', 'w') as f:
    json.dump({k: {wk: {h: {kk: (float(vv) if isinstance(vv, (int, float, np.floating)) else vv)
                            for kk, vv in v.items()} for h, v in w.items()}
                   for wk, w in v.items()} for k, v in results.items()}, f, indent=1)
print("\nsaved scripts/miner2_reval_20340616.json")
