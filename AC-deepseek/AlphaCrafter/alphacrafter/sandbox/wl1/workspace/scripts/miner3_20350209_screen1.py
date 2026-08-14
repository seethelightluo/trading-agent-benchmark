"""miner_3 2035-02-09: screen candidate factor ideas on the fresh panel.
Evaluates 1d/5d/10d forward Spearman IC; admission gates |ic1|>=0.0070, |icir1|>=0.0840.
Reports library max correlation for redundancy audit.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PANEL = 'scripts/panel_cache_20350209.pkl'
with open(PANEL, 'rb') as f:
    panel = pd.read_pickle(f)

px = panel['close']; ret = panel['ret']
hi = panel['high']; lo = panel['low']; op = panel['open']; vol = panel['vol']

# ---------- library reconstruction (for rho audit) ----------
lib = {}
for n in [1, 2, 3, 5]:
    lib[f'rev_{n}d'] = -(np.log(px) - np.log(px.shift(n)))
    rmax = px.rolling(n).max(); rmin = px.rolling(n).min()
    lib[f'nclv_{n}d'] = -(px - rmin) / (rmax - rmin)
lib['id_rev_1d'] = -(px / px.shift(1) - 1.0)
lib['nbody_1d'] = -((px - op) / (hi - lo))
lib['rev_1d_vs'] = -(np.log(px) - np.log(px.shift(1))) / ret.rolling(20).std()
lib['mom_120d_skip5'] = px.shift(5) / px.shift(125) - 1.0
lib['vol_of_vol20x60'] = ret.rolling(20).std().rolling(60).std()
vix = panel['macro']['VIX'].reindex(px.index).ffill()
vix_ret = vix.pct_change()
betas = pd.DataFrame(index=px.index, columns=px.columns, dtype=float)
for i in range(60, len(ret)):
    a = ret.iloc[i-60:i]; b = vix_ret.iloc[i-60:i]
    m = a.notna() & b.notna()
    if int(m.sum().sum()) < 10:
        continue
    aa = a[m]; bb = b[m]
    cov = (aa * bb).mean() - aa.mean() * bb.mean()
    var = bb.var()
    if var > 0:
        betas.iloc[i] = cov / var
vix_trend = vix_ret.rolling(20).mean()
lib['vix_beta_cond_60x20'] = betas * np.sign(vix_trend).values[:, None]

# ---------- candidates ----------
clv = (px - lo) / (hi - lo).replace(0, np.nan)
cands = {}
cands['clv_20d'] = clv.rolling(20).mean()                      # close-location avg 20d
cands['clv_20d_rev'] = -clv.rolling(20).mean()                 # negated
cands['vol_ratio_5_60'] = ret.rolling(5).std() / ret.rolling(60).std()   # vol term structure
cands['vol_ratio_10_60'] = ret.rolling(10).std() / ret.rolling(60).std()
cands['amihud_20d'] = (ret.abs() / vol.replace(0, np.nan)).rolling(20).mean()  # illiquidity
cands['downside_ratio_20d'] = ret.clip(upper=0).rolling(20).std() / ret.rolling(20).std()
cands['range_std_20d'] = ((hi - lo) / px).rolling(20).std()
cands['hl_avg_20d'] = ((hi - lo) / px).rolling(20).mean()
cands['gap_mom_10d'] = (op / px.shift(1) - 1).rolling(10).sum()
cands['skew_20d'] = ret.rolling(20).skew()
cands['drawdown_60d'] = px / px.rolling(60).max() - 1.0
cands['updown_ratio_20d'] = ret.clip(lower=0).rolling(20).mean() / (-ret.clip(upper=0).rolling(20).mean())
cands['mom_accel_10_60'] = (px / px.shift(10) - 1) - (px / px.shift(60) - 1)
cands['vol_adj_range_20'] = ((hi - lo) / px).rolling(20).mean() / ret.rolling(20).std()
cands['zscore_20d'] = (px - px.rolling(20).mean()) / px.rolling(20).std()
cands['corr_btc_20d'] = ret.rolling(20).corr(ret['BTC'])
cands['corr_btc_60d'] = ret.rolling(60).corr(ret['BTC'])
cands['rngpos_trend_20'] = clv.rolling(20).mean() - clv.rolling(5).mean()  # CLV drift

def eval_cand(fac, h=1, min_valid=8):
    fwd = px.pct_change(h).shift(-h)
    ics = {}
    for dt in fac.index:
        f = fac.loc[dt]; r = fwd.loc[dt]
        m = f.notna() & r.notna()
        if int(m.sum()) >= min_valid:
            rho, _ = spearmanr(f[m], r[m])
            ics[dt] = rho
    s = pd.Series(ics)
    if len(s) < 30:
        return {'n': len(s)}
    icm = s.mean(); icstd = s.std()
    return {'n': int(len(s)), 'ic': float(icm), 'icir': float(icm/icstd) if icstd > 0 else np.nan,
            'hit': float((np.sign(s) == np.sign(icm)).mean()),
            'cov': float(fac.notna().sum().sum() / (fac.shape[0]*fac.shape[1]))}

def rho_lib(fac):
    alld = fac.index.intersection(lib['rev_1d'].index)
    rho_max, anchor = 0.0, None
    for k, v in lib.items():
        vv = v.loc[alld]
        mm = fac.loc[alld].notna() & vv.notna()
        if int(mm.sum().sum()) < 200:
            continue
        a = fac.loc[alld][mm].values.flatten(); b = vv[mm].values.flatten()
        rho = np.corrcoef(a, b)[0, 1]
        if abs(rho) > abs(rho_max):
            rho_max, anchor = rho, k
    return float(rho_max), anchor

G_IC, G_ICIR = 0.0070, 0.0840
rows = []
for name, fac in cands.items():
    r1 = eval_cand(fac, 1)
    r5 = eval_cand(fac, 5)
    r10 = eval_cand(fac, 10)
    rm, anc = rho_lib(fac)
    if 'ic' in r1:
        gate = abs(r1['ic']) >= G_IC and abs(r1['icir']) >= G_ICIR
        rows.append((name, r1, r5, r10, rm, anc, gate))
        print(f"{name:18s} h1 IC={r1['ic']:+.4f} ICIR={r1['icir']:+.3f} hit={r1['hit']:.3f} n={r1['n']:4d} cov={r1['cov']:.2f} | "
              f"h5 IC={r5.get('ic', float('nan')):+.4f} ICIR={r5.get('icir', float('nan')):+.3f} | "
              f"h10 IC={r10.get('ic', float('nan')):+.4f} ICIR={r10.get('icir', float('nan')):+.3f} | "
              f"rho_max={rm:.3f}({anc}) GATE={'PASS' if gate else 'fail'}")
    else:
        print(f"{name:18s} insufficient n={r1['n']}")

print("\nSummary of gates (1d):")
for name, r1, r5, r10, rm, anc, gate in rows:
    if gate:
        print(f"  PASS: {name} ic={r1['ic']:+.4f} icir={r1['icir']:+.3f} rho_max={rm:.3f}({anc})")
