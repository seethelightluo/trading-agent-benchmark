"""miner_3 2035-03-09: screen candidate factor ideas on fresh panel (through 2035-03-08).
Vectorized rank-IC. Admission gates |ic1|>=0.0070, |icir1|>=0.0840.
"""
import numpy as np
import pandas as pd

PANEL = 'scripts/panel_cache_20350309.pkl'
with open(PANEL, 'rb') as f:
    panel = pd.read_pickle(f)

px = panel['close']; ret = panel['ret']
hi = panel['high']; lo = panel['low']; op = panel['open']; vol = panel['vol']
macro = panel['macro']

# ---------- library reconstruction (vectorized) ----------
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
vix = macro['VIX'].reindex(px.index).ffill()
vix_ret = vix.pct_change()
# vectorized 60d beta: cov(asset, vix_ret)/var(vix_ret)
cov60 = ret.rolling(60).cov(vix_ret)
var60 = vix_ret.rolling(60).var()
betas = cov60 / var60.replace(0, np.nan)
vix_trend = vix_ret.rolling(20).mean()
lib['vix_beta_cond_60x20'] = betas * np.sign(vix_trend).values[:, None]

# ---------- candidates ----------
clv = (px - lo) / (hi - lo).replace(0, np.nan)
park = np.sqrt(252 * np.log(hi / lo).pow(2).rolling(20).mean())
vol20 = ret.rolling(20).std()

cands = {}
cands['clv_20d_rev'] = -clv.rolling(20).mean()
cands['vol_ratio_5_60'] = ret.rolling(5).std() / ret.rolling(60).std()
cands['amihud_20d'] = (ret.abs() / vol.replace(0, np.nan)).rolling(20).mean()
cands['downside_ratio_20d'] = ret.clip(upper=0).rolling(20).std() / ret.rolling(20).std()
cands['range_std_20d'] = ((hi - lo) / px).rolling(20).std()
cands['gap_mom_10d'] = (op / px.shift(1) - 1).rolling(10).sum()
cands['skew_20d'] = ret.rolling(20).skew()
cands['drawdown_60d'] = px / px.rolling(60).max() - 1.0
cands['mom_accel_10_60'] = (px / px.shift(10) - 1) - (px / px.shift(60) - 1)
cands['zscore_20d'] = (px - px.rolling(20).mean()) / px.rolling(20).std()
cands['corr_btc_20d'] = ret.rolling(20).corr(ret['BTC'])
cands['rngpos_trend_20'] = clv.rolling(20).mean() - clv.rolling(5).mean()
cands['mom_20d_voladj'] = (px / px.shift(20) - 1) / vol20
cands['mom_60d_voladj'] = (px / px.shift(60) - 1) / vol20
cands['mom_120d_voladj'] = (px / px.shift(120) - 1) / vol20
cands['mom_20d_skip5_voladj'] = (px.shift(5) / px.shift(25) - 1) / vol20
cands['risk_adj_rev_5d'] = -(px / px.shift(5) - 1) / vol20
cands['park_ratio_20'] = park / vol20
cands['park_20d'] = park
cands['hl_range_z_20'] = ((hi - lo) / px).rolling(20).mean() / ((hi - lo) / px).rolling(20).std()
body = (px - op) / px
cands['body_mean_10d'] = body.rolling(10).mean()
cands['body_neg_frac_20d'] = (body < 0).rolling(20).mean()
cands['up_gap_frac_20d'] = (op > px.shift(1)).rolling(20).mean()
cands['tail_ratio_20d'] = ((op - lo) / (hi - op).replace(0, np.nan)).rolling(20).mean()
dxy = macro['DXY'].reindex(px.index).ffill()
dxy_ret = dxy.pct_change()
cands['dxy_beta_60d'] = ret.rolling(60).cov(dxy_ret) / dxy_ret.rolling(60).var().replace(0, np.nan)
usd = macro['USDJPY'].reindex(px.index).ffill()
usd_ret = usd.pct_change()
cands['usdjpy_beta_60d'] = ret.rolling(60).cov(usd_ret) / usd_ret.rolling(60).var().replace(0, np.nan)
cands['corr_xau_20d'] = ret.rolling(20).corr(ret['XAU'])
cands['corr_us10y_20d'] = ret.rolling(20).corr(ret['US10Y'])
cands['corr_spx_20d'] = ret.rolling(20).corr(ret['SPX'])
cands['vix_level_z'] = (vix - vix.rolling(120).mean()) / vix.rolling(120).std()
vma20 = vol.rolling(20).mean().replace(0, np.nan)
cands['vol_ratio_5_20'] = vol.rolling(5).mean() / vma20
cands['vol_z_20'] = (vol.rolling(5).mean() - vma20) / vol.rolling(20).std()
cands['vol_price_corr_20'] = vol.rolling(20).corr(px)

# ---------- vectorized rank IC ----------
def rank_ic_series(fac, fwd, min_valid=8):
    fr = fac.rank(axis=1)
    rr = fwd.rank(axis=1)
    valid = fr.notna() & rr.notna()
    n = valid.sum(axis=1)
    fr = fr.where(valid); rr = rr.where(valid)
    fr_c = fr.sub(fr.mean(axis=1), axis=0)
    rr_c = rr.sub(rr.mean(axis=1), axis=0)
    num = (fr_c * rr_c).sum(axis=1)
    den = np.sqrt((fr_c ** 2).sum(axis=1) * (rr_c ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    ic = ic.where(n >= min_valid)
    return ic

def stats(ss):
    ss = ss.dropna()
    if len(ss) < 30:
        return {'n': int(len(ss))}
    icm = ss.mean(); icstd = ss.std()
    return {'n': int(len(ss)), 'ic': float(icm), 'icir': float(icm/icstd) if icstd > 0 else np.nan,
            'hit': float((np.sign(ss) == np.sign(icm)).mean())}

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
    cov = float(fac.notna().sum().sum() / (fac.shape[0]*fac.shape[1]))
    out = {}
    for h in [1, 5, 10]:
        fwd = px.pct_change(h).shift(-h)
        ic = rank_ic_series(fac, fwd)
        out[h] = stats(ic)
        if h == 1:
            icr = ic[ic.index >= ic.index.max() - pd.Timedelta(days=365)]
            out['rec1y'] = stats(icr)
    rm, anc = rho_lib(fac)
    r1, r5, r10 = out[1], out[5], out[10]
    rec = out['rec1y']
    if 'ic' in r1:
        gate = abs(r1['ic']) >= G_IC and abs(r1['icir']) >= G_ICIR
        rows.append((name, r1, r5, r10, rec, rm, anc, gate))
        print(f"{name:26s} h1 IC={r1['ic']:+.4f} ICIR={r1['icir']:+.3f} hit={r1['hit']:.3f} n={r1['n']:4d} cov={cov:.2f} | "
              f"h5 IC={r5.get('ic', float('nan')):+.4f} ICIR={r5.get('icir', float('nan')):+.3f} | "
              f"h10 IC={r10.get('ic', float('nan')):+.4f} ICIR={r10.get('icir', float('nan')):+.3f} | "
              f"rec1y IC={rec.get('ic', float('nan')):+.4f} ICIR={rec.get('icir', float('nan')):+.3f} | "
              f"rho_max={rm:.3f}({anc}) GATE={'PASS' if gate else 'fail'}")
    else:
        print(f"{name:26s} insufficient n={r1['n']}")

print("\nSummary of gates (1d):")
for name, r1, r5, r10, rec, rm, anc, gate in rows:
    if gate:
        print(f"  PASS: {name} ic={r1['ic']:+.4f} icir={r1['icir']:+.3f} rho_max={rm:.3f}({anc})")
