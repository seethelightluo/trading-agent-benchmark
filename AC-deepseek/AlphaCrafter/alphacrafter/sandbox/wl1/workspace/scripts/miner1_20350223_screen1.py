"""miner_1 2035-02-23: library re-validation + new candidate screen on panel thru 2035-02-22.
Gates: |ic1| >= 0.0070, |icir1| >= 0.0840 (same-horizon admission metric = 1d).
Reports full-sample, live-window (>=2026-07-16), and recent-250d IC for timeliness.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PANEL = 'scripts/panel_cache_20350223.pkl'
with open(PANEL, 'rb') as f:
    panel = pd.read_pickle(f)

px = panel['close']; ret = panel['ret']
hi = panel['high']; lo = panel['low']; op = panel['open']

# ---------- library reconstruction ----------
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

# ---------- new candidates ----------
spx = ret['SPX']
xau = ret['XAU']
cands = {}

# momentum / trend family
cands['mom_10d'] = px / px.shift(10) - 1.0
cands['mom_20d'] = px / px.shift(20) - 1.0
cands['mom_60d'] = px / px.shift(60) - 1.0
cands['mom_20d_voladj'] = (px / px.shift(20) - 1.0) / ret.rolling(20).std()
cands['mom_60d_voladj'] = (px / px.shift(60) - 1.0) / ret.rolling(60).std()
cands['trend_persist_20_60'] = np.where((px/px.shift(20)-1 < 0) & (px/px.shift(60)-1 < 0), -1.0, 1.0)  # persistent downtrend flag (neg = de-rank)
cands['ma20_slope'] = (px.rolling(20).mean() / px.rolling(20).mean().shift(5) - 1.0)
cands['zscore_60d'] = (px - px.rolling(60).mean()) / px.rolling(60).std()
cands['above_ma20'] = np.where(px > px.rolling(20).mean(), 1.0, -1.0)

# defensive / cross-asset beta family (crisis regime, VIX>60, XAU defensive working)
def rolling_beta(a, b, win):
    out = pd.DataFrame(index=a.index, columns=a.columns, dtype=float)
    for i in range(win, len(a)):
        aa = a.iloc[i-win:i]; bb = b.iloc[i-win:i]
        m = aa.notna() & bb.notna()
        if int(m.sum().sum()) < win * 0.5:
            continue
        av = aa[m].values; bv = bb[m].values
        var = bv.var()
        if var > 1e-14:
            cov = (av * bv).mean() - av.mean() * bv.mean()
            out.iloc[i] = cov / var
    return out

cands['spx_beta_60d'] = rolling_beta(ret, spx, 60)
cands['spx_beta_20d'] = rolling_beta(ret, spx, 20)
cands['xau_corr_60d'] = ret.rolling(60).corr(xau)
cands['xau_beta_60d'] = rolling_beta(ret, xau, 60)
# resilience: avg return on days SPX down > 1% (last 60d) - negative = fragile
spx_down = (spx < -0.01)
cands['downside_resil_60d'] = ret[spx_down].rolling(60, min_periods=5).mean()
# defensive score: low SPX beta + high XAU corr
cands['defensive_score'] = -cands['spx_beta_60d'].clip(-3, 3) + cands['xau_corr_60d']

# macro-conditional momentum
cands['dxy_mom_20d'] = (panel['macro']['DXY'].reindex(px.index).ffill() / panel['macro']['DXY'].reindex(px.index).ffill().shift(20) - 1.0)
cands['dxy_mom_60d'] = (panel['macro']['DXY'].reindex(px.index).ffill() / panel['macro']['DXY'].reindex(px.index).ffill().shift(60) - 1.0)
cands['vix_mom_20d'] = (vix / vix.shift(20) - 1.0)
cands['usdjpy_mom_20d'] = (panel['macro']['USDJPY'].reindex(px.index).ffill() / panel['macro']['USDJPY'].reindex(px.index).ffill().shift(20) - 1.0)

# reversal refinements
cands['dd_60d'] = px / px.rolling(60).max() - 1.0
cands['dd_60d_voladj'] = (px / px.rolling(60).max() - 1.0) / ret.rolling(60).std()
cands['rev_5d_voladj'] = -(px / px.shift(5) - 1.0) / ret.rolling(20).std()
cands['stoch_20d_rev'] = -(px - px.rolling(20).min()) / (px.rolling(20).max() - px.rolling(20).min())

# range / vol structure
cands['hl_pct_20d'] = ((hi - lo) / px).rolling(20).mean()
cands['range_expand_5_60'] = ((hi - lo) / px).rolling(5).mean() / ((hi - lo) / px).rolling(60).mean()
cands['vol_ratio_20_60'] = ret.rolling(20).std() / ret.rolling(60).std()

def eval_cand(fac, h=1, min_valid=8, start=None, end=None):
    fwd = px.pct_change(h).shift(-h)
    ics = {}
    for dt in fac.index:
        if start is not None and dt < start:
            continue
        if end is not None and dt > end:
            continue
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
LIVE = pd.Timestamp('2026-07-16')
RECENT = px.index.max() - pd.Timedelta(days=400)

print("=" * 110)
print("LIBRARY RE-VALIDATION (full sample thru 2035-02-22)")
print("=" * 110)
for name, fac in lib.items():
    r1 = eval_cand(fac, 1)
    if 'ic' in r1:
        gate = abs(r1['ic']) >= G_IC and abs(r1['icir']) >= G_ICIR
        print(f"{name:20s} h1 IC={r1['ic']:+.4f} ICIR={r1['icir']:+.3f} hit={r1['hit']:.3f} n={r1['n']:4d} cov={r1['cov']:.2f} GATE={'PASS' if gate else 'fail'}")
    else:
        print(f"{name:20s} insufficient n={r1['n']}")

print()
print("=" * 110)
print("NEW CANDIDATES (full sample thru 2035-02-22; gates on 1d)")
print("=" * 110)
rows = []
for name, fac in cands.items():
    r1 = eval_cand(fac, 1)
    r5 = eval_cand(fac, 5)
    r10 = eval_cand(fac, 10)
    rlive = eval_cand(fac, 1, start=LIVE)
    rrec = eval_cand(fac, 1, start=RECENT)
    rm, anc = rho_lib(fac)
    if 'ic' in r1:
        gate = abs(r1['ic']) >= G_IC and abs(r1['icir']) >= G_ICIR
        rows.append((name, r1, r5, r10, rlive, rrec, rm, anc, gate))
        print(f"{name:22s} h1 IC={r1['ic']:+.4f} ICIR={r1['icir']:+.3f} hit={r1['hit']:.3f} n={r1['n']:4d} cov={r1['cov']:.2f} | "
              f"h5 IC={r5.get('ic', float('nan')):+.4f}/ICIR={r5.get('icir', float('nan')):+.3f} | "
              f"h10 IC={r10.get('ic', float('nan')):+.4f}/ICIR={r10.get('icir', float('nan')):+.3f} | "
              f"live1 IC={rlive.get('ic', float('nan')):+.4f}(n={rlive.get('n', 0)}) rec250 IC={rrec.get('ic', float('nan')):+.4f}(n={rrec.get('n', 0)}) | "
              f"rho_max={rm:.3f}({anc}) GATE={'PASS' if gate else 'fail'}")
    else:
        print(f"{name:22s} insufficient n={r1['n']}")

print("\n--- PASSING (1d gate) ---")
for name, r1, r5, r10, rlive, rrec, rm, anc, gate in rows:
    if gate:
        print(f"  PASS: {name} ic={r1['ic']:+.4f} icir={r1['icir']:+.3f} hit={r1['hit']:.3f} rho_max={rm:.3f}({anc}) live={rlive.get('ic', float('nan')):+.4f} rec250={rrec.get('ic', float('nan')):+.4f}")

print("\n--- borderline (|ic|>=0.005 OR |icir|>=0.05) ---")
for name, r1, r5, r10, rlive, rrec, rm, anc, gate in rows:
    if not gate and (abs(r1['ic']) >= 0.005 or abs(r1['icir']) >= 0.05):
        print(f"  NEAR: {name} ic={r1['ic']:+.4f} icir={r1['icir']:+.3f} hit={r1['hit']:.3f} rho_max={rm:.3f}({anc})")
