"""miner_2 cycle 2028-04-06: fresh candidate screen on 15-asset cross-asset universe.
ASOF = visible_through 2028-04-05. Gates: |IC|>=0.0070, |ICIR|>=0.0840 at H=10
(Spearman rank IC, >=8 assets/date). Frozen assets (trailing 60d zero-vol) -> NaN.
Regime: deepening risk-off (VIX ~22 live), 4/15 frozen feeds (000688/SOX/NDX/CN10Y),
negative trends, XAU leadership, crypto collapsed. Focus: time-series structure of
returns (autocorrelation/persistence), intraday-vs-overnight structure, vol-spike
regime momentum, macro-beta conditionals (EURUSD/USDJPY).
"""
import json, os, zlib, base64, io, time
import numpy as np
import pandas as pd

t0 = time.time()
ASOF = '2028-04-05'
H = 10
WATCH = ['000300.SH','000688.SH','BTC','CN10Y','COPPER','ETH','HSI','N225','NDX','SOX','SPX','SX5E','US10Y','WTI','XAU']
MACRO = {'VIX':'../persistent/index_data/VIX.csv','DXY':'../persistent/index_data/DXY.csv',
         'USDCNY':'../persistent/index_data/USDCNY.csv','USDJPY':'../persistent/index_data/USDJPY.csv',
         'EURUSD':'../persistent/index_data/EURUSD.csv'}

px = pd.DataFrame({s: pd.read_csv(f'../persistent/stock_data/{s}.csv').pipe(
        lambda d: d.assign(date=pd.to_datetime(d['date'])).set_index('date')['close'].loc[:pd.Timestamp(ASOF)])
    for s in WATCH}).sort_index()
# also load open for overnight/intraday split
opn = pd.DataFrame({s: pd.read_csv(f'../persistent/stock_data/{s}.csv').pipe(
        lambda d: d.assign(date=pd.to_datetime(d['date'])).set_index('date')['open'].loc[:pd.Timestamp(ASOF)])
    for s in WATCH}).sort_index()
hi = pd.DataFrame({s: pd.read_csv(f'../persistent/stock_data/{s}.csv').pipe(
        lambda d: d.assign(date=pd.to_datetime(d['date'])).set_index('date')['high'].loc[:pd.Timestamp(ASOF)])
    for s in WATCH}).sort_index()
lo = pd.DataFrame({s: pd.read_csv(f'../persistent/stock_data/{s}.csv').pipe(
        lambda d: d.assign(date=pd.to_datetime(d['date'])).set_index('date')['low'].loc[:pd.Timestamp(ASOF)])
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

def build(fn):
    out = pd.DataFrame({s: fn(px[s]) for s in WATCH}).sort_index()
    # void frozen assets: trailing 60d zero-vol -> NaN for all candidates on those dates
    z = ret1.rolling(60, min_periods=30).std().replace(0, np.nan)
    return out.where(z.notna())

vix = macro['VIX'].reindex(px.index)
dxy = macro['DXY'].reindex(px.index)
usdcny = macro['USDCNY'].reindex(px.index)
usdjpy = macro['USDJPY'].reindex(px.index)
eurusd = macro['EURUSD'].reindex(px.index)
us10 = px['US10Y']

vix_r = vix.pct_change()
vix_chg20 = vix / vix.shift(20) - 1.0
us10_chg20 = us10 / us10.shift(20) - 1.0

factors = {}

# A1: ac_5x20 - rolling autocorrelation of 5d (weekly) returns over 20 windows (trend persistence)
def _ac(s, k=5, w=20):
    r = s.pct_change(k)
    v = r.dropna()
    out = v.rolling(w, min_periods=8).apply(lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= 8 else np.nan, raw=False)
    return out.reindex(s.index)
factors['ac_5x20'] = build(lambda s: _ac(s, 5, 20))

# A2: sign_persist_10 - sign autocorrelation of daily returns over 10d (streak-ish but distinct)
def _sign_ac(s, w=10):
    r = np.sign(s.pct_change()).dropna()
    out = r.rolling(w, min_periods=6).apply(lambda x: (x.iloc[:-1] == x.iloc[1:]).mean() - 0.5 if len(x) >= 6 else np.nan, raw=False)
    return out.reindex(s.index)
factors['sign_persist_10'] = build(lambda s: _sign_ac(s, 10))

# A3: overnight_ratio_20 - overnight vol (open vs prev close) / intraday vol (close vs open), 20d
def _on_ratio(s, w=20):
    o = opn[s.name]; c = px[s.name]
    onr = o / c.shift(1) - 1.0
    intr = c / o - 1.0
    onv = onr.rolling(w, min_periods=10).std()
    inv = intr.rolling(w, min_periods=10).std()
    return (onv / inv.replace(0, np.nan)).reindex(s.index)
factors['overnight_ratio_20'] = build(_on_ratio)

# A4: range_eff_20 - Parkinson range vol / close-close vol (gap/efficiency structure)
def _range_eff(s, w=20):
    h = hi[s.name]; l = lo[s.name]; c = px[s.name]
    rng = np.log(h / l).dropna()
    rv = rng.rolling(w, min_periods=10).std()
    cv = ret1[s.name].rolling(w, min_periods=10).std()
    return (rv / cv.replace(0, np.nan)).reindex(s.index)
factors['range_eff_20'] = build(_range_eff)

# A5: vol_ratio_5x60 - short/long vol ratio (vol spike state)
factors['vol_ratio_5x60'] = build(lambda s: rstd(s, 5) / rstd(s, 60).replace(0, np.nan))

# A6: days_since_high_60 - log duration since 60d high (drawdown duration)
def _days_high(s, w=60):
    c = s.dropna()
    roll_max = c.rolling(w, min_periods=30).max()
    idx = np.arange(len(c))
    # days since the rolling max was achieved: use expanding argmax within window via reindex trick
    def _dsh(x):
        mx = np.max(x)
        pos = np.where(x == mx)[0]
        return (len(x) - 1 - pos[-1]) if len(pos) else np.nan
    out = c.rolling(w, min_periods=30).apply(_dsh, raw=True)
    return np.log1p(out).reindex(s.index)
factors['days_since_high_60'] = build(_days_high)

# A7: up_frac_40 - fraction of positive daily returns over 40d (trend consistency)
factors['up_frac_40'] = build(lambda s: (s.pct_change() > 0).rolling(40, min_periods=20).mean())

# A8: volspike_mom_20x5 - 20d momentum gated on vol spike (short vol > long vol)
def f8(s):
    m = s / s.shift(20) - 1.0
    gate = (rstd(s, 5) / rstd(s, 60).replace(0, np.nan)) > 1.0
    return m.where(gate)
factors['volspike_mom_20'] = build(f8)

# A9: eurusd_beta_60 - 60d beta vs EURUSD (global risk appetite exposure)
factors['eurusd_beta_60'] = build(lambda s: rbeta(s, eurusd, 60))

# A10: usdjpy_beta_60 - 60d beta vs USDJPY (carry/risk exposure)
factors['usdjpy_beta_60'] = build(lambda s: rbeta(s, usdjpy, 60))

# A11: vixhigh_mom_10x5 - 10d momentum gated on high VIX level (> 60th pct of trailing year)
vix_pct60 = vix.rolling(252, min_periods=60).quantile(0.6)
vix_high = vix > vix_pct60
def f11(s):
    m = s / s.shift(10) - 1.0
    return m.where(vix_high.reindex(s.index))
factors['vixhigh_mom_10'] = build(f11)

# A12: spx_beta_down_60 - downside beta: beta of asset to SPX on SPX-down days (defensive tilt)
spx = px['SPX']
spx_down = spx.pct_change() < 0
factors['spx_beta_down_60'] = build(lambda s: rbeta(s, spx, 60, cond=spx_down))

# ---------- library panels ----------
lib_panels = {}
lib_panels['mom_10d_skip5'] = retk(15) - retk(5)
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
    fr = fz.rank(axis=1); rr = fwd.rank(axis=1)
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
    icd_r = icd[icd.index >= icd.index[-1] - pd.Timedelta(days=365)] if len(icd) else icd
    st_r = ic_stats(icd_r)
    ranks = fz.rank(axis=1)
    to = ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1)
    cov = fz.notna().sum().sum() / (len(fz) * len(WATCH))
    rhos = {ln: spearman_panel_rho(fz, lp.reindex(fz.index)) for ln, lp in lib_panels.items()}
    maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
    decay = {hh: round(ic_stats(cs_ic_vec(fz, fwd_panel(hh)))['ic'], 4) for hh in [1, 2, 3, 5, 10, 20]}
    reg = {}
    for lab, m in [('2020-2021', icd.index < pd.Timestamp('2022-01-01')),
                   ('2022-2023', (icd.index >= pd.Timestamp('2022-01-01')) & (icd.index < pd.Timestamp('2024-01-01'))),
                   ('2024+', icd.index >= pd.Timestamp('2024-01-01'))]:
        sub = icd[m]
        if len(sub):
            ss = ic_stats(sub)
            reg[lab] = [round(ss['ic'], 4), round(ss['icir'], 4), int(ss['n_dates'])]
    gate = (abs(st['ic']) >= 0.0070) and (abs(st['icir']) >= 0.0840)
    rho_ok = maxrho < 0.5
    flag = 'PASS' if (gate and rho_ok) else ('GATE_PASS_RHO_HI' if gate else 'fail')
    res[name] = {'ic': st['ic'], 'icir': st['icir'], 'hit': st['hit'], 'n_dates': st['n_dates'],
                 'avg_assets': st['avg_n'], 'recent1y_ic': st_r['ic'], 'recent1y_icir': st_r['icir'],
                 'turnover_10d': to, 'coverage': cov,
                 'rho_lib': rhos, 'max_lib_rho': maxrho, 'decay': decay, 'regime': reg, 'flag': flag}
    print(f"== {name} == {flag}  ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_a={st['avg_n']:.1f}")
    print(f"   recent1y ic={st_r['ic']:.4f} icir={st_r['icir']:.4f} | to10={to:.3f} cov={cov:.3f} maxrho={maxrho:.3f}")
    print(f"   decay={decay}")
    print(f"   regime={reg}")

json.dump(res, open('scripts/_miner2_20280406_batchA_results.json', 'w'), indent=1, default=str)
print(f"\nSaved scripts/_miner2_20280406_batchA_results.json  ({time.time()-t0:.0f}s)")
