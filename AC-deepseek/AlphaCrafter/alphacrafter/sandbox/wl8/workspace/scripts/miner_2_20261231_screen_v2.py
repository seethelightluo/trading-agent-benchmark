"""miner_2 cycle 2026-12-31: screen new candidate factors on the 15-asset cross-asset universe.
Panel-A construction (audit-equivalent): full union calendar, NaN gaps preserved (like usdcny_beta_60 artifact).
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 at H=10, Spearman, >=8 valid assets/date, dates with >=8 valid.
Library rho vs ensemble factors (mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20, usdcny_beta_60).
"""
import json, os, zlib, base64, io
import numpy as np
import pandas as pd

ASOF = '2026-12-30'
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
    """rolling beta of y on x over w rows on the union calendar (NaN-gap aware via cov of valid pairs)."""
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
    return pd.DataFrame({s: fn(px[s]) for s in WATCH}).sort_index()


# macro series on px index
vix = macro['VIX'].reindex(px.index)
dxy = macro['DXY'].reindex(px.index)
usdcny = macro['USDCNY'].reindex(px.index)
usdjpy = macro['USDJPY'].reindex(px.index)
us10 = px['US10Y']; cn10 = px['CN10Y']; cn = px['000300.SH']

vix_up_2pct = vix.pct_change() > 0.02
dxy_up = dxy.pct_change() > 0
ushike20 = (us10 / us10.shift(20) - 1.0) > 0

factors = {}

# F1: csi300_hedged_mom_60 - residual momentum vs CN (asset mom - beta*CN mom)
cn_mom60 = retk(65)[cn].dropna()
def f1(p):
    b = rbeta(p, cn, 60)
    return retk(65)[p.name] - b * cn_mom60.reindex(px.index)
# build manually per asset to preserve name
def f1b(s):
    b = rbeta(px[s], cn, 60)
    m = retk(65)[s]
    return m - b * cn_mom60
factors['csi300_hedged_mom_60'] = build(f1b)

# F2: vol_ratio_30x120 - 30d RV / 120d RV (vol acceleration; expect NEGATIVE IC in risk-off regime)
def f2(s):
    return rstd(px[s], 30) / rstd(px[s], 120)
factors['vol_ratio_30x120'] = build(f2)

# F3: range_pos_60h - (close - low_60)/(high_60 - low_60) 60d range position (mean-reversion candidate)
def f3(s):
    v = px[s].dropna()
    hi = v.rolling(60, min_periods=30).max()
    lo = v.rolling(60, min_periods=30).min()
    return ((v - lo) / (hi - lo)).reindex(px.index)
factors['range_pos_60h'] = build(f3)

# F4: dxy_hedged_mom_60 - 60d momentum residual vs DXY beta (USD-orthogonal trend)
dy = dxy.pct_change()
def f4(s):
    b = rbeta(px[s], dxy, 60)
    dxy_mom = (dxy / dxy.shift(60) - 1.0)
    return retk(65)[s] - b * dxy_mom
factors['dxy_hedged_mom_60'] = build(f4)

# F5: rate_spread_cond_mom_40 - 40d mom * sign((US10Y-CN10Y) 40d change) (DM/EM rate-spread regime)
spread = us10 - cn10
sp_chg = spread / spread.shift(40) - 1.0
def f5(s):
    return retk(45)[s] * np.sign(sp_chg)
factors['rate_spread_cond_mom_40'] = build(f5)

# F6: vix_rise_semi_vol_ratio_20 - 20d downside semi-vol on VIX-up days / 20d total RV
def f6(s):
    r = px[s].pct_change()
    m = vix_up_2pct.reindex(px.index).fillna(False)
    down = r.where(m & (r < 0))
    semi = down.rolling(20, min_periods=6).std()
    tot = r.rolling(20, min_periods=10).std()
    return (semi / tot).reindex(px.index)
factors['vix_rise_semi_vol_ratio_20'] = build(f6)

# F7: vol_confirm_mom_20 - 20d mom * sign(20d RV - 60d RV) (momentum + volume/vol confirmation)
def f7(s):
    m20 = retk(20)[s]
    vol_z = rstd(px[s], 20) - rstd(px[s], 60)
    return m20 * np.sign(vol_z)
factors['vol_confirm_mom_20'] = build(f7)

# F8: cn10y_led_mom_40 - 40d mom * sign(CN10Y 40d change) (CN rate cycle)
cn10_chg40 = cn10 / cn10.shift(40) - 1.0
def f8(s):
    return retk(45)[s] * np.sign(cn10_chg40)
factors['cn10y_led_mom_40'] = build(f8)

# ---------- library panels ----------
lib_panels = {}
lib_panels['mom_10d_skip5'] = retk(15)
vix_r = vix.pct_change()
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame(
    {s: rbeta(px[s], vix, 60, cond=(vix / vix.shift(20) - 1.0) > 0) for s in WATCH}).sort_index()
lib_panels['yield_beta_cond_60x20'] = pd.DataFrame(
    {s: rbeta(px[s], us10, 60, cond=ushike20) for s in WATCH}).sort_index()
d = json.load(open('factors/usdcny_beta_60.json'))
txt = zlib.decompress(base64.b64decode(d['validation']['signal_artifact']['data'])).decode()
ua = pd.read_csv(io.StringIO(txt)).set_index('date')
ua.index = pd.to_datetime(ua.index)
lib_panels['usdcny_beta_60'] = ua.reindex(px.index)

# ---------- evaluation ----------
def cross_sectional_ic(fpanel, fwd_panel, min_assets=8):
    recs = []
    common = fpanel.index.intersection(fwd_panel.index)
    for dd in common:
        f = fpanel.loc[dd]; r = fwd_panel.loc[dd]
        m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        n = int(m.sum())
        if n >= min_assets:
            ic = f[m].corr(r[m], method='spearman')
            if not np.isnan(ic):
                recs.append((dd, ic, n))
    return pd.DataFrame(recs, columns=['date', 'ic', 'n']).set_index('date')

def ic_stats(icdf):
    if len(icdf) == 0:
        return {'ic': np.nan, 'icir': np.nan, 'hit': np.nan, 'n_dates': 0, 'avg_n': 0}
    ic = icdf['ic'].mean(); sd = icdf['ic'].std(ddof=1)
    icir = ic / sd if sd and not np.isnan(sd) and sd > 0 else 0.0
    return {'ic': ic, 'icir': icir, 'hit': (icdf['ic'] > 0).mean(), 'n_dates': len(icdf), 'avg_n': icdf['n'].mean()}

def spearman_panel_rho(a, b):
    common = a.index.intersection(b.index)
    rhos = []
    for dd in common:
        av = a.loc[dd]; bv = b.loc[dd]
        m = av.notna() & bv.notna() & np.isfinite(av) & np.isfinite(bv)
        if m.sum() >= 6:
            r = av[m].corr(bv[m], method='spearman')
            if not np.isnan(r):
                rhos.append(r)
    return float(np.mean(rhos)) if rhos else np.nan

def fwd_panel(h):
    return (px.shift(-h) / px - 1.0)

res = {}
for name, f in factors.items():
    fz = f.replace([np.inf, -np.inf], np.nan)
    icd = cross_sectional_ic(fz, fwd_panel(H))
    st = ic_stats(icd)
    ranks = fz.rank(axis=1)
    to = ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1)
    cov = fz.notna().sum().sum() / (len(fz) * len(WATCH))
    rhos = {ln: spearman_panel_rho(fz, lp.reindex(fz.index)) for ln, lp in lib_panels.items()}
    maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
    decay = {hh: round(ic_stats(cross_sectional_ic(fz, fwd_panel(hh)))['ic'], 4) for hh in [1, 2, 3, 5, 10, 20]}
    gate = (abs(st['ic']) >= 0.0070) and (abs(st['icir']) >= 0.0840)
    flag = 'PASS' if gate else 'fail'
    res[name] = {'ic': st['ic'], 'icir': st['icir'], 'hit': st['hit'], 'n_dates': st['n_dates'],
                 'avg_assets': st['avg_n'], 'turnover_10d': to, 'coverage': cov,
                 'rho_lib': rhos, 'max_lib_rho': maxrho, 'decay': decay, 'flag': flag}
    print(f"== {name} == {flag}  ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_a={st['avg_n']:.1f}")
    print(f"   to10={to:.3f} cov={cov:.3f} maxrho={maxrho:.3f} rhos={ {k: (None if v!=v else round(v,3)) for k,v in rhos.items()} }")
    print(f"   decay={decay}")

json.dump(res, open('scripts/_miner2_cycle25_screen_results.json', 'w'), indent=1, default=str)
print("\nSaved scripts/_miner2_cycle25_screen_results.json")