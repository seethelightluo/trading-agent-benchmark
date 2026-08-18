"""miner_2 cycle 2027-04-08 batch B: orthogonalized / low-collinearity candidates.
ASOF = visible_through 2027-04-07. Gates: |IC|>=0.0070, |ICIR|>=0.0840 at H=10 (Spearman, >=8 assets/date).
Frozen assets (trailing 60d zero-vol) -> NaN. Focus: diversify the momentum-heavy ensemble
(mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20, usdcny_beta_60).
"""
import json, os, zlib, base64, io, time
import numpy as np
import pandas as pd

t0 = time.time()
ASOF = '2027-04-07'
H = 10
WATCH = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
MACRO = {'VIX':'../persistent/index_data/VIX.csv','DXY':'../persistent/index_data/DXY.csv',
         'USDCNY':'../persistent/index_data/USDCNY.csv','USDJPY':'../persistent/index_data/USDJPY.csv',
         'EURUSD':'../persistent/index_data/EURUSD.csv'}

px = pd.DataFrame({s: pd.read_csv(f'../persistent/stock_data/{s}.csv').pipe(
        lambda d: d.assign(date=pd.to_datetime(d['date'])).set_index('date')['close'].loc[:pd.Timestamp(ASOF)])
    for s in WATCH}).sort_index()
macro = pd.DataFrame({m: pd.read_csv(p).pipe(
        lambda d: d.assign(date=pd.to_datetime(d['date'])).set_index('date')['close'].loc[:pd.Timestamp(ASOF)])
    for m, p in MACRO.items()}).sort_index()

ret1 = px.pct_change()
retk = lambda k: px / px.shift(k) - 1.0

def rstd(s, w, minp=None):
    v = s.dropna()
    if minp is None:
        minp = max(3, int(w * 0.5))
    return v.rolling(w, min_periods=minp).std().reindex(px.index)

def rbeta(y, x, w, cond=None, minp=None):
    df = pd.concat([y.rename('y'), x.rename('x')], axis=1)
    if cond is not None:
        c = cond.reindex(df.index).fillna(False).astype(bool)
        df = df.where(c)
    if minp is None:
        minp = max(6, int(w * 0.4))
    cov = df['y'].rolling(w, min_periods=minp).cov(df['x'])
    var = df['x'].rolling(w, min_periods=minp).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan)

def build(fn):
    out = pd.DataFrame({s: fn(px[s]) for s in WATCH}).sort_index()
    z = ret1.rolling(60, min_periods=30).std().replace(0, np.nan)
    return out.where(z.notna())

vix = macro['VIX'].reindex(px.index)
dxy = macro['DXY'].reindex(px.index)
usdcny = macro['USDCNY'].reindex(px.index)
usdjpy = macro['USDJPY'].reindex(px.index)
us10 = px['US10Y']; cn10 = px['CN10Y']

vix_r = vix.pct_change()
vix_chg20 = vix / vix.shift(20) - 1.0
us10_chg20 = us10 / us10.shift(20) - 1.0
mom10 = retk(15)  # mom_10d_skip5 proxy (ret from t-15..t-5 per its definition: close/shift(15)-1 ?? use same as ens)

factors = {}

# B1: reversal_10 orth - 10d reversal residualized on mom_10d_skip5 via rolling beta (lower collinearity)
def f1(s):
    rv = -(s / s.shift(10) - 1.0)
    b = rbeta(rv, mom10[s], 60)
    return rv - b * mom10[s]
factors['reversal_10_orth'] = build(f1)

# B2: drawup_40 - distance below 40d high (deeper pullback, longer reversion horizon)
factors['drawup_40'] = build(lambda s: -(s / s.rolling(40, min_periods=24).max() - 1.0))

# B3: corridor_20x60 - 20d range / 60d range (volatility compression ratio; low = coiled)
def f3(s):
    hi20 = s.rolling(20, min_periods=12).max(); lo20 = s.rolling(20, min_periods=12).min()
    hi60 = s.rolling(60, min_periods=30).max(); lo60 = s.rolling(60, min_periods=30).min()
    return ((hi20 - lo20) / (hi60 - lo60)).reindex(s.index)
factors['corridor_20x60'] = build(f3)

# B4: min_gain_15 - most negative daily close-to-close print over 15d (lottery-loss / capitulation marker)
factors['min_gain_15'] = build(lambda s: s.pct_change().rolling(15, min_periods=8).min())

# B5: dd_speed_60 - 60d max drawdown intensity (maxdd scaled by drawdown duration, faster dd = worse)
def f5(s):
    v = s.rolling(60, min_periods=30).max()
    dd = s / v - 1.0
    mdd = dd.rolling(60, min_periods=30).min()
    return mdd.reindex(s.index)
factors['maxdd_60'] = build(f5)

# B6: skew_level_60 - 60d realized skew (negative skew = crash-prone)
factors['skew_level_60'] = build(lambda s: s.pct_change().rolling(60, min_periods=36).skew())

# B7: hsar_break_20 - time-decayed proximity to recent low (mean-reversion timing: days since 40d low)
def f7(s):
    v = s.rolling(40, min_periods=24).min()
    return (s / v - 1.0).reindex(s.index)
factors['dist_lo_40'] = build(f7)

# B8: vixchg_beta_cond_60x20 - beta of asset to VIX %change when VIX rising * sign (regime hedge demand)
factors['vixchg_beta_cond'] = pd.DataFrame(
    {s: rbeta(px[s], vix_r, 60, cond=vix_chg20 > 0) for s in WATCH}).sort_index()
factors['vixchg_beta_cond'] = factors['vixchg_beta_cond'].where(
    ret1.rolling(60, min_periods=30).std().replace(0, np.nan).notna())

# ---------- library panels ----------
lib_panels = {}
lib_panels['mom_10d_skip5'] = mom10
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame(
    {s: rbeta(px[s], vix, 60, cond=vix_chg20 > 0) for s in WATCH}).sort_index()
lib_panels['yield_beta_cond_60x20'] = pd.DataFrame(
    {s: rbeta(px[s], us10, 60, cond=us10_chg20 > 0) for s in WATCH}).sort_index()
d = json.load(open('factors/usdcny_beta_60.json'))
txt = zlib.decompress(base64.b64decode(d['validation']['signal_artifact']['data'])).decode()
ua = pd.read_csv(io.StringIO(txt)).set_index('date')
ua.index = pd.to_datetime(ua.index)
lib_panels['usdcny_beta_60'] = ua.reindex(px.index)

print(f"[data] px dates={len(px)} range={px.index[0]}..{px.index[-1]}")

# ---------- evaluation ----------
def cs_ic_vec(fz, fwd):
    fr = fz.rank(axis=1); rr = fwd.rank(axis=1)
    common = fz.index.intersection(fwd.index)
    ff = fr.loc[common]; gg = rr.loc[common]; fv = fz.loc[common]; gv = fwd.loc[common]
    mask = fv.notna() & gv.notna() & np.isfinite(fv.values) & np.isfinite(gv.values)
    n = mask.sum(axis=1); ok = n >= 8
    fa = ff.where(mask).sub(ff.where(mask).mean(axis=1), axis=0)
    ga = gg.where(mask).sub(gg.where(mask).mean(axis=1), axis=0)
    num = (fa * ga).sum(axis=1)
    den = np.sqrt((fa ** 2).sum(axis=1) * (ga ** 2).sum(axis=1))
    ic = (num / den.replace(0, np.nan)).where(ok)
    return pd.DataFrame({'ic': ic, 'n': n}, index=common).dropna(subset=['ic'])

def ic_stats(icdf):
    if len(icdf) == 0:
        return {'ic': np.nan, 'icir': np.nan, 'hit': np.nan, 'n_dates': 0, 'avg_n': 0}
    ic = icdf['ic'].mean(); sd = icdf['ic'].std(ddof=1)
    icir = ic / sd if sd and not np.isnan(sd) and sd > 0 else 0.0
    return {'ic': ic, 'icir': icir, 'hit': (icdf['ic'] > 0).mean(), 'n_dates': len(icdf), 'avg_n': icdf['n'].mean()}

def spearman_panel_rho(a, b):
    common = a.index.intersection(b.index)
    if len(common) == 0:
        return np.nan
    ar = a.loc[common].rank(axis=1); br = b.loc[common].rank(axis=1)
    av = a.loc[common]; bv = b.loc[common]
    mask = av.notna() & bv.notna()
    n = mask.sum(axis=1); ok = n >= 6
    aa = ar.where(mask).sub(ar.where(mask).mean(axis=1), axis=0)
    bb = br.where(mask).sub(br.where(mask).mean(axis=1), axis=0)
    num = (aa * bb).sum(axis=1); den = np.sqrt((aa ** 2).sum(axis=1) * (bb ** 2).sum(axis=1))
    r = (num / den.replace(0, np.nan)).where(ok).dropna()
    return float(r.mean()) if len(r) else np.nan

def fwd_panel(h):
    return px.shift(-h) / px - 1.0

res = {}
for name, f in factors.items():
    fz = f.replace([np.inf, -np.inf], np.nan)
    icd = cs_ic_vec(fz, fwd_panel(H))
    st = ic_stats(icd)
    ranks = fz.rank(axis=1)
    to = ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1)
    cov = fz.notna().sum().sum() / (len(fz) * len(WATCH))
    rhos = {ln: spearman_panel_rho(fz, lp.reindex(fz.index)) for ln, lp in lib_panels.items()}
    maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
    decay = {hh: round(ic_stats(cs_ic_vec(fz, fwd_panel(hh)))['ic'], 4) for hh in [1, 2, 3, 5, 10, 20]}
    gate = (abs(st['ic']) >= 0.0070) and (abs(st['icir']) >= 0.0840)
    rho_ok = maxrho < 0.5
    flag = 'PASS' if (gate and rho_ok) else ('GATE_PASS_RHO_HI' if gate else 'fail')
    res[name] = {'ic': st['ic'], 'icir': st['icir'], 'hit': st['hit'], 'n_dates': st['n_dates'],
                 'avg_assets': st['avg_n'], 'turnover_10d': to, 'coverage': cov,
                 'rho_lib': rhos, 'max_lib_rho': maxrho, 'decay': decay, 'flag': flag}
    print(f"== {name} == {flag}  ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_a={st['avg_n']:.1f}")
    print(f"   to10={to:.3f} cov={cov:.3f} maxrho={maxrho:.3f} rhos={ {k: (None if v!=v else round(v,3)) for k,v in rhos.items()} }")
    print(f"   decay={decay}")

json.dump(res, open('scripts/_miner2_cycle29_screenB_results.json', 'w'), indent=1, default=str)
print(f"\nSaved scripts/_miner2_cycle29_screenB_results.json  ({time.time()-t0:.0f}s)")