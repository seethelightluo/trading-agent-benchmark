"""miner_2 cycle 2027-03-25: fresh candidate screen on 15-asset cross-asset universe.
ASOF = visible_through 2027-03-24. Gates: |IC|>=0.0070, |ICIR|>=0.0840 at H=10 (Spearman, >=8 assets/date).
Library rho vs usdcny_beta_60 (decoded artifact) + ensemble recomputes (mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20).
Rho threshold 0.5 flags eviction risk.
"""
import json, os, zlib, base64, io, time
import numpy as np
import pandas as pd

t0 = time.time()
ASOF = '2027-03-24'
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
    """rolling R^2 of linear fit to time index (trend quality)."""
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
    return pd.DataFrame({s: fn(px[s]) for s in WATCH}).sort_index()

vix = macro['VIX'].reindex(px.index)
dxy = macro['DXY'].reindex(px.index)
usdcny = macro['USDCNY'].reindex(px.index)
usdjpy = macro['USDJPY'].reindex(px.index)
us10 = px['US10Y']; cn10 = px['CN10Y']; cn = px['000300.SH']

vix_r = vix.pct_change()

factors = {}

# F1: usdcny_orthogonal_mom_60 - 60d momentum residual after removing USDCNY-beta exposure
# (explicitly orthogonal to the active library factor usdcny_beta_60)
cnymom60 = (usdcny / usdcny.shift(60) - 1.0)
def f1(s):
    b = rbeta(px[s], usdcny, 60)
    return retk(65)[s] - b * cnymom60
factors['usdcny_orthogonal_mom_60'] = build(f1)

# F2: vix_orthogonal_mom_60 - 60d momentum residual after removing VIX-beta exposure
vixmom60 = (vix / vix.shift(60) - 1.0)
def f2(s):
    b = rbeta(px[s], vix, 60)
    return retk(65)[s] - b * vixmom60
factors['vix_orthogonal_mom_60'] = build(f2)

# F3: skew_shift_20x60 - change in 20d realized skew vs 60d (skew dynamics, not level)
def f3(s):
    r = px[s].pct_change()
    sk20 = r.rolling(20, min_periods=12).skew()
    sk60 = r.rolling(60, min_periods=30).skew()
    return (sk20 - sk60).reindex(px.index)
factors['skew_shift_20x60'] = build(f3)

# F4: vol_term_10x60 - short/long vol term structure (10d RV / 60d RV)
def f4(s):
    return rstd(px[s], 10, minp=5) / rstd(px[s], 60, minp=30)
factors['vol_term_10x60'] = build(f4)

# F5: range_pos_10x40 - 10d range position (mean-reversion candidate, short-horizon)
def f5(s):
    v = px[s].dropna()
    hi = v.rolling(10, min_periods=6).max()
    lo = v.rolling(10, min_periods=6).min()
    return ((v - lo) / (hi - lo)).reindex(px.index)
factors['range_pos_10x40'] = build(f5)

# F6: max_gain_20 - max single-day gain over 20d (lottery/tail preference)
def f6(s):
    r = px[s].pct_change()
    return r.rolling(20, min_periods=12).max().reindex(px.index)
factors['max_gain_20'] = build(f6)

# F7: idio_vol_z_60 - idiosyncratic vol vs SPX (60d residual vol z-scored cross-sectionally later; here raw)
spx = px['SPX']
def f7(s):
    b = rbeta(px[s], spx, 60)
    res = ret1[s] - b * ret1['SPX']
    return res.rolling(60, min_periods=30).std().reindex(px.index)
factors['idio_vol_60'] = build(f7)

# F8: yield_cycle_mom_60 - 60d momentum * sign(US10Y 60d change) (rate-cycle conditional momentum)
us10_chg60 = (us10 / us10.shift(60) - 1.0)
def f8(s):
    return retk(65)[s] * np.sign(us10_chg60)
factors['yield_cycle_mom_60'] = build(f8)

# F9: trend_quality_30 - R^2 of 30d linear fit (trend quality)
def f9(s):
    return r2fit(px[s], 30)
factors['trend_quality_30'] = build(f9)

# ---------- library panels ----------
lib_panels = {}
lib_panels['mom_10d_skip5'] = retk(15)
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame(
    {s: rbeta(px[s], vix, 60, cond=(vix / vix.shift(20) - 1.0) > 0) for s in WATCH}).sort_index()
lib_panels['yield_beta_cond_60x20'] = pd.DataFrame(
    {s: rbeta(px[s], us10, 60, cond=(us10 / us10.shift(20) - 1.0) > 0) for s in WATCH}).sort_index()
d = json.load(open('factors/usdcny_beta_60.json'))
txt = zlib.decompress(base64.b64decode(d['validation']['signal_artifact']['data'])).decode()
ua = pd.read_csv(io.StringIO(txt)).set_index('date')
ua.index = pd.to_datetime(ua.index)
lib_panels['usdcny_beta_60'] = ua.reindex(px.index)

print(f"[data] px dates={len(px)} range={px.index[0]}..{px.index[-1]}  macro dates={len(macro)}")

# ---------- evaluation (vectorized rank-based) ----------
def rank_panel(fpanel):
    return fpanel.rank(axis=1)

def cs_ic_vec(fz, fwd):
    """Per-date Spearman IC via cross-sectional ranks + pearson on ranks."""
    fr = fz.rank(axis=1)
    rr = fwd.rank(axis=1)
    common = fz.index.intersection(fwd.index)
    ff = fr.loc[common]; gg = rr.loc[common]; fv = fz.loc[common]; gv = fwd.loc[common]
    mask = fv.notna() & gv.notna() & np.isfinite(fv.values) & np.isfinite(gv.values)
    n = mask.sum(axis=1)
    ok = n >= 8
    # demeaned ranks
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

json.dump(res, open('scripts/_miner2_cycle28_screen_results.json', 'w'), indent=1, default=str)
print(f"\nSaved scripts/_miner2_cycle28_screen_results.json  ({time.time()-t0:.0f}s)")