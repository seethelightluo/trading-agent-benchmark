"""miner_2 cycle 2027-04-08: fresh candidate screen on 15-asset cross-asset universe.
ASOF = visible_through 2027-04-07 (current date 2027-04-08). Gates: |IC|>=0.0070, |ICIR|>=0.0840
at H=10 (Spearman rank IC, >=8 assets/date). Frozen assets (zero-std trailing window) -> NaN.
Library rho vs usdcny_beta_60 (decoded artifact) + ensemble recomputes (mom_10d_skip5,
vix_beta_cond_60x20, yield_beta_cond_60x20). Regime context: 3 consecutive negative blocks,
VIX re-expansion, momentum whipsaw -> focus on mean-reversion / vol-state conditional ideas.
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

def rcorr(s, x, w, minp=None):
    df = pd.concat([s.dropna().rename('a'), x.dropna().rename('b')], axis=1).dropna()
    if minp is None:
        minp = max(4, int(w * 0.5))
    return df['a'].rolling(w, min_periods=minp).corr(df['b']).reindex(px.index)

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

def r2fit(s, w, minp=None):
    v = s.dropna()
    if minp is None:
        minp = max(6, int(w * 0.4))
    n = len(v)
    out = pd.Series(np.nan, index=v.index)
    x = np.arange(w, dtype=float)
    for i in range(w - 1, n):
        seg = v.iloc[i - w + 1:i + 1]
        if seg.notna().sum() < minp:
            continue
        y = seg.values.astype(float)
        if np.std(y) == 0 or not np.isfinite(y).all():
            continue
        c = np.corrcoef(x[:len(y)], y)[0, 1]
        out.iloc[i] = c * c
    return out.reindex(px.index)

def build(fn):
    out = pd.DataFrame({s: fn(px[s]) for s in WATCH}).sort_index()
    # void frozen assets: trailing 60d zero-vol -> NaN for all candidates on those dates
    z = ret1.rolling(60, min_periods=30).std().replace(0, np.nan)
    return out.where(z.notna())

vix = macro['VIX'].reindex(px.index)
dxy = macro['DXY'].reindex(px.index)
usdcny = macro['USDCNY'].reindex(px.index)
usdjpy = macro['USDJPY'].reindex(px.index)
us10 = px['US10Y']; cn10 = px['CN10Y']; cn = px['000300.SH']

vix_r = vix.pct_change()
vix_chg20 = vix / vix.shift(20) - 1.0
vix_chg60 = vix / vix.shift(60) - 1.0
us10_chg10 = us10 / us10.shift(10) - 1.0
us10_chg20 = us10 / us10.shift(20) - 1.0

factors = {}

# R1: reversal_10 - plain 10d reversal (negative short momentum)
factors['reversal_10'] = build(lambda s: -(s / s.shift(10) - 1.0))

# R2: vol_adj_reversal_20 - 20d reversal scaled by 20d vol (contrarian, vol-normalized)
factors['vol_adj_reversal_20'] = build(lambda s: -(s / s.shift(20) - 1.0) / rstd(s, 20))

# R3: zscore_20 - distance from 20d mean in std units (negative = oversold)
factors['zscore_20'] = build(lambda s: ((s - s.rolling(20, min_periods=12).mean()) / rstd(s, 20)))

# R4: drawup_20 - distance below 20d high (pullback depth, long-side reversion)
factors['drawup_20'] = build(lambda s: -(s / s.rolling(20, min_periods=12).max() - 1.0))

# R5: rsi_7 - classic 7d RSI (oversold low values -> long)
def _rsi(s, w=7):
    r = s.pct_change()
    up = r.clip(lower=0).rolling(w, min_periods=4).mean()
    dn = (-r.clip(upper=0)).rolling(w, min_periods=4).mean()
    rs = up / dn.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).reindex(s.index)
factors['rsi_7'] = build(lambda s: _rsi(s, 7))

# R6: vixstate_reversal_10 - 10d reversal, downweighted/hidden when VIX not expanding (state gate: vix 20d chg>0 else NaN)
def f6(s):
    rv = -(s / s.shift(10) - 1.0)
    gate = (vix_chg20 > 0).reindex(rv.index)
    return rv.where(gate)
factors['vixstate_reversal_10'] = build(f6)

# R7: volterm_reversal_20 - 20d reversal conditioned on short vol > long vol (vol spike regime)
def f7(s):
    rv = -(s / s.shift(20) - 1.0)
    vt = rstd(s, 5) / rstd(s, 60)
    return rv.where(vt > 1.0)
factors['volterm_reversal_20'] = build(f7)

# R8: slowdown_20x5 - momentum deceleration: 5d ret minus 20d trend (whipsaw/slowdown signal)
factors['slowdown_20x5'] = build(lambda s: (s / s.shift(5) - 1.0) - (s / s.shift(20) - 1.0))

# R9: trend_quality_60 - R^2 of 60d linear fit (high quality trend continuation)
factors['trend_quality_60'] = build(lambda s: r2fit(s, 60))

# R10: yield_reversal_20 - reversal in yield-sensitive complex: asset 20d ret * -sign(US10Y 20d chg)
def f10(s):
    return -(s / s.shift(20) - 1.0) * np.sign(us10_chg20)
factors['yield_reversal_20'] = build(f10)

# R11: usdjpy_reversal_cond - reversal in USDJPY-sensitive assets when JPY weakening (risk-on reversion)
usdjpy_chg20 = usdjpy / usdjpy.shift(20) - 1.0
def f11(s):
    rv = -(s / s.shift(10) - 1.0)
    return rv.where(usdjpy_chg20 > 0)
factors['usdjpy_reversal_10'] = build(f11)

# R12: max_gain_10x40 - recent max daily gain low = no lottery blowoff (contrarian quality)
factors['min_gain_10'] = build(lambda s: s.pct_change().rolling(10, min_periods=6).min())

# R13: skew_shift_10x60 - skew rising (negative skew deepening -> crash risk, short)
def f13(s):
    r = s.pct_change()
    sk10 = r.rolling(10, min_periods=6).skew()
    sk60 = r.rolling(60, min_periods=30).skew()
    return (sk10 - sk60).reindex(s.index)
factors['skew_shift_10x60'] = build(f13)

# R14: dxy_reversal_cond - reversal of assets when DXY falling (risk-on regime reversion)
dxy_chg20 = dxy / dxy.shift(20) - 1.0
def f14(s):
    rv = -(s / s.shift(10) - 1.0)
    return rv.where(dxy_chg20 < 0)
factors['dxy_reversal_10'] = build(f14)

# ---------- library panels ----------
lib_panels = {}
lib_panels['mom_10d_skip5'] = retk(15)
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame(
    {s: rbeta(px[s], vix, 60, cond=vix_chg20 > 0) for s in WATCH}).sort_index()
lib_panels['yield_beta_cond_60x20'] = pd.DataFrame(
    {s: rbeta(px[s], us10, 60, cond=us10_chg20 > 0) for s in WATCH}).sort_index()
d = json.load(open('factors/usdcny_beta_60.json'))
txt = zlib.decompress(base64.b64decode(d['validation']['signal_artifact']['data'])).decode()
ua = pd.read_csv(io.StringIO(txt)).set_index('date')
ua.index = pd.to_datetime(ua.index)
lib_panels['usdcny_beta_60'] = ua.reindex(px.index)

print(f"[data] px dates={len(px)} range={px.index[0]}..{px.index[-1]}  macro dates={len(macro)}")

# ---------- evaluation ----------
def cs_ic_vec(fz, fwd):
    fr = fz.rank(axis=1)
    rr = fwd.rank(axis=1)
    common = fz.index.intersection(fwd.index)
    ff = fr.loc[common]; gg = rr.loc[common]; fv = fz.loc[common]; gv = fwd.loc[common]
    mask = fv.notna() & gv.notna() & np.isfinite(fv.values) & np.isfinite(gv.values)
    n = mask.sum(axis=1)
    ok = n >= 8
    fa = ff.where(mask).sub(ff.where(mask).mean(axis=1), axis=0)
    ga = gg.where(mask).sub(gg.where(mask).mean(axis=1), axis=0)
    num = (fa * ga).sum(axis=1)
    den = np.sqrt((fa ** 2).sum(axis=1) * (ga ** 2).sum(axis=1))
    ic = num / den.replace(0, np.nan)
    ic = ic.where(ok)
    out = pd.DataFrame({'ic': ic, 'n': n}, index=common)
    return out.dropna(subset=['ic'])

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
    n = mask.sum(axis=1)
    ok = n >= 6
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

json.dump(res, open('scripts/_miner2_cycle29_screen_results.json', 'w'), indent=1, default=str)
print(f"\nSaved scripts/_miner2_cycle29_screen_results.json  ({time.time()-t0:.0f}s)")