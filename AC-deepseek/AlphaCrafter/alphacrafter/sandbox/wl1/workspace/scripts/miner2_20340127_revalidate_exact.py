"""miner2 2034-01-27: revalidate existing factor library through 2034-01-12 with EXACT library definitions."""
import pandas as pd, numpy as np, json, time

t0 = time.time()
panel = pd.read_pickle('scripts/panel_cache_20340113.pkl')
close = panel['close']; ret = panel['ret']; vol = panel['vol']
open_ = panel['open']; high = panel['high']; low = panel['low']
macro = panel['macro']
print(f"panel loaded {time.time()-t0:.1f}s", flush=True)

def ic_series_vec(signal, close_, hz=1):
    sig_r = signal.rank(axis=1, pct=True).values
    fwd = (np.log(close_.shift(-hz) / close_)).values
    dates = signal.index
    ics = []
    for t in range(len(signal) - hz):
        s = sig_r[t]; f = fwd[t]
        m = np.isfinite(s) & np.isfinite(f)
        if m.sum() >= 8:
            ic = np.corrcoef(s[m], f[m])[0, 1]
            if np.isfinite(ic):
                ics.append(ic)
    return pd.Series(ics, index=dates[:len(ics)])

def stats(ics, signal):
    if len(ics) == 0:
        return dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
    sd = float(ics.std(ddof=1))
    return dict(ic=float(ics.mean()), icir=float(ics.mean()/sd) if sd > 0 else np.nan,
                hit=float((ics > 0).mean()), n=len(ics))

# ---- exact library definitions ----
sig = {}
# rev family: -(ln(close_t) - ln(close_{t-k}))
for k in [1, 2, 3, 5]:
    sig[f'rev_{k}d'] = -(np.log(close) - np.log(close.shift(k)))
# nclv family: -(close - rolling_min(low,k)) / (rolling_max(high,k) - rolling_min(low,k))
def nclv(k):
    hi = high.rolling(k).max(); lo = low.rolling(k).min()
    return -(close - lo) / (hi - lo).replace(0, np.nan)
for k in [1, 2, 3, 5]:
    sig[f'nclv_{k}d'] = nclv(k)
# id_rev_1d: -(close/open - 1)
sig['id_rev_1d'] = -(close/open_ - 1.0)
# nbody_1d: -(close - open)/(high - low)
sig['nbody_1d'] = -(close - open_) / (high - low).replace(0, np.nan)
# rev_1d_vs: -(ln(close_t)-ln(close_{t-1}))/rolling_std(ret,20)
sig['rev_1d_vs'] = -(np.log(close) - np.log(close.shift(1))) / ret.rolling(20).std().replace(0, np.nan)
# mom_120d_skip5: close.shift(5)/close.shift(125) - 1
sig['mom_120d_skip5'] = close.shift(5) / close.shift(125) - 1.0
# vol_of_vol20x60: std(pct_change,20).rolling(60).std()
sig['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
# vix_beta_cond_60x20: -beta(asset_ret, VIX_ret, 60) * (VIX/VIX.shift(20)-1)
vix = macro['VIX']
vix_r = np.log(vix).diff()
def vix_beta(win=60):
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for c in close.columns:
        a = ret[c]; b = vix_r
        out[c] = a.rolling(win).cov(b) / b.rolling(win).var()
    return out
vb = vix_beta(60)
sig['vix_beta_cond_60x20'] = -vb * (vix / vix.shift(20) - 1.0).reindex(close.index).values.reshape(-1, 1)
print(f"signals built {time.time()-t0:.1f}s", flush=True)

results = {}
GATE_IC, GATE_ICIR = 0.0070, 0.0840
print(f"\nGates: |IC|>={GATE_IC} |ICIR|>={GATE_ICIR}\n")
for name, s in sig.items():
    s = s.reindex(close.index)
    res = {}
    for label, sl in [('full', slice(None)), ('recent_2y', close.index[-520:]), ('recent_1y', close.index[-260:])]:
        ss = s.loc[sl]
        res[label] = {h: stats(ic_series_vec(ss, close.loc[sl], h), ss) for h in [1, 2, 3, 5, 10]}
    results[name] = res
    f1 = res['full'][1]; f10 = res['full'][10]; r2 = res['recent_2y'][1]; r1 = res['recent_1y'][1]
    cov = float(s.notna().mean().mean())
    rp = s.rank(axis=1, pct=True)
    turn = float(rp.diff().abs().mean().mean())
    gate = "PASS" if (abs(f1['ic']) >= GATE_IC and abs(f1['icir']) >= GATE_ICIR) else "fail"
    print(f"{name:20s} [{gate}] FULL ic1={f1['ic']:.4f} icir1={f1['icir']:.3f} hit={f1['hit']:.3f} n={f1['n']:4d} "
          f"ic10={f10['ic']:.4f} icir10={f10['icir']:.3f} | 2Y ic1={r2['ic']:.4f} icir1={r2['icir']:.3f} "
          f"| 1Y ic1={r1['ic']:.4f} icir1={r1['icir']:.3f} | cov={cov:.3f} turn={turn:.3f}", flush=True)

with open('scripts/miner2_reval_20340127.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print(f"\nsaved scripts/miner2_reval_20340127.json in {time.time()-t0:.1f}s")
