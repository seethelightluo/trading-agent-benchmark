"""miner2 2032-04-23: optimized revalidation of effective library factors."""
import pandas as pd, numpy as np, json
from scipy.stats import rankdata

panel = pd.read_pickle('scripts/panel_cache_20320423.pkl')
close = panel['close']; ret = panel['ret']; vol = panel['vol']; macro = panel['macro']
print("panel:", close.shape, close.index.min().date(), "->", close.index.max().date(), flush=True)

def fwd_ret(h):
    return close.shift(-h) / close - 1.0

def row_rank(m):
    """Row-wise rank (average method), NaN preserved."""
    out = np.full(m.shape, np.nan)
    for i in range(m.shape[0]):
        row = m[i]
        msk = ~np.isnan(row)
        if msk.sum() > 0:
            r = rankdata(row[msk])
            out[i, msk] = r
    return out

def ic_series(sig_rank, fwd_rank, min_n=8):
    ics, dates = [], []
    for i in range(sig_rank.shape[0]):
        s = sig_rank[i]; f = fwd_rank[i]
        m = ~(np.isnan(s) | np.isnan(f))
        if m.sum() < min_n:
            continue
        ic = np.corrcoef(s[m], f[m])[0, 1]
        if np.isfinite(ic):
            ics.append(ic); dates.append(i)
    return np.array(ics), dates

def eval_factor(signal, close_sub, horizons=(1, 2, 3, 5, 10), min_n=8):
    out = {}
    sig_rank = row_rank(signal.values)
    for h in horizons:
        fwd = fwd_ret(h).reindex(signal.index)
        fr = row_rank(fwd.values)
        ics, dates = ic_series(sig_rank, fr, min_n)
        if len(ics) == 0:
            out[h] = dict(ic=np.nan, icir=np.nan, hit=np.nan, n=0)
        else:
            ic = float(np.mean(ics)); sd = float(np.std(ics, ddof=1))
            out[h] = dict(ic=ic, icir=ic/sd if sd > 0 else np.nan,
                          hit=float(np.mean(ics > 0)), n=len(ics))
    rp = signal.rank(axis=1, pct=True)
    out['coverage'] = float(signal.notna().mean().mean())
    out['turnover_1d_rank'] = float(rp.diff().abs().mean().mean())
    out['n_dates'] = int(signal.shape[0])
    return out

# ---- build signals ----
sig = {}
for k in [1, 2, 3, 5]:
    sig[f'rev_{k}d'] = -(np.log(close) - np.log(close.shift(k)))
def nclv(k):
    hi = close.rolling(k).max(); lo = close.rolling(k).min()
    lv = (close - lo) / (hi - lo).replace(0, np.nan)
    return -lv
for k in [1, 2, 3, 5]:
    sig[f'nclv_{k}d'] = nclv(k)
sig['id_rev_1d'] = -(np.log(close) - np.log(panel['open']))
body = np.log(close) - np.log(panel['open'])
sig['nbody_1d'] = -body.abs() * np.sign(body)
sig['rev_1d_vs'] = -(np.log(close) - np.log(close.shift(1))) * (vol / vol.rolling(20).mean()).clip(0.5, 2.0)
sig['mom_120d_skip5'] = np.log(close) - np.log(close.shift(125))
sig['mom_10d_skip5'] = np.log(close) - np.log(close.shift(15))
rv20 = ret.rolling(20).std(); rv60 = ret.rolling(60).std()
sig['vol_of_vol20x60'] = -(rv20.rolling(60).std())
vix = macro['VIX']; vix_r = np.log(vix).diff()
def vix_beta(win=60):
    out = pd.DataFrame(index=close.index, columns=close.columns, dtype=float)
    for c in close.columns:
        a = ret[c]; b = vix_r
        out[c] = a.rolling(win).cov(b) / b.rolling(win).var()
    return out
vb = vix_beta(60)
cond = (vix > vix.rolling(120).median()).astype(float)
sig['vix_beta_cond_60x20'] = -vb * cond.shift(1)

# ---- evaluate ----
results = {}
for name, s in sig.items():
    s = s.reindex(close.index)
    res = {}
    for label, sl in [('full', slice(None)), ('recent_2y', close.index[-520:]), ('recent_1y', close.index[-260:])]:
        res[label] = eval_factor(s.loc[sl], close.loc[sl])
    results[name] = res

for name, res in results.items():
    h1 = res['full'].get(1, {}); r2 = res['recent_2y'].get(1, {}); r1 = res['recent_1y'].get(1, {})
    print(f"{name:18s} FULL ic1={h1.get('ic',np.nan):.4f} icir1={h1.get('icir',np.nan):.3f} hit={h1.get('hit',np.nan):.3f} "
          f"| 2Y ic1={r2.get('ic',np.nan):.4f} icir1={r2.get('icir',np.nan):.3f} "
          f"| 1Y ic1={r1.get('ic',np.nan):.4f} icir1={r1.get('icir',np.nan):.3f} "
          f"| cov={res['full'].get('coverage',np.nan):.3f} turn={res['full'].get('turnover_1d_rank',np.nan):.3f}", flush=True)

with open('scripts/miner2_reval_20320423.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved scripts/miner2_reval_20320423.json")
