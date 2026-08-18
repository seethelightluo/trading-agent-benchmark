"""miner_3 cycle 2028-02-24 (ASOF 2028-02-23): broad cross-asset factor screen.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 at H=10 on 15-asset cross-section (>=8 valid/date).
Also revalidates usdcny_beta_60 (the only active library factor).
Library for rho: usdcny_beta_60 (recomputed).
NOTE: 000688/CN10Y/NDX/SOX frozen in worldline (flat recent closes) -> reported.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, load_prices, load_macro, cross_sectional_ic, ic_stats, spearman_panel_rho, regime_split

ASOF = '2028-02-23'
H = 10

px = load_prices(ASOF)
macro = load_macro(ASOF)
INDEX = px.index

def vseries(s):
    return s.dropna()

def ret1(s):
    return vseries(s).pct_change().reindex(INDEX)

def retk(s, k):
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)

def rstd(s, w, minp=None):
    v = vseries(s)
    if minp is None:
        minp = max(3, int(w * 0.5))
    return v.rolling(w, min_periods=minp).std().reindex(INDEX)

def rcorr(s, x, w, minp=None):
    vs, vx = vseries(s), vseries(x)
    df = pd.concat([vs.rename('a'), vx.rename('b')], axis=1, sort=True).dropna()
    if minp is None:
        minp = max(4, int(w * 0.5))
    return df['a'].rolling(w, min_periods=minp).corr(df['b']).reindex(INDEX)

def rbeta(y, x, w, cond=None, minp=None):
    vy, vx = vseries(y), vseries(x)
    df = pd.concat([vy.rename('y'), vx.rename('x')], axis=1, sort=True).dropna()
    if cond is not None:
        c = cond.reindex(df.index).fillna(False).astype(bool)
    else:
        c = pd.Series(True, index=df.index)
    ym, xm = df['y'].where(c), df['x'].where(c)
    if minp is None:
        minp = max(6, int(w * 0.4))
    cov = ym.rolling(w, min_periods=minp).cov(xm)
    var = xm.rolling(w, min_periods=minp).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan).reindex(INDEX)

def build(name, fn):
    cols = {}
    for s in WATCH:
        try:
            cols[s] = fn(s, px[s])
        except Exception:
            cols[s] = np.nan
    return pd.DataFrame(cols).sort_index()

us10 = px['US10Y']
cn10y = px['CN10Y']
vix = macro['VIX']
dxy = macro['DXY']
usdjpy = macro['USDJPY']
usdcny = macro['USDCNY']
eurusd = macro['EURUSD']

factors = {}

# --- Library revalidation: usdcny_beta_60 ---
factors['usdcny_beta_60'] = build('usdcny_beta_60', lambda s, p: rbeta(p, usdcny, 60))

# --- Family A: cross-asset beta ---
factors['vix_beta_60'] = build('vix_beta_60', lambda s, p: rbeta(p, vix, 60))
factors['usdjpy_beta_60'] = build('usdjpy_beta_60', lambda s, p: rbeta(p, usdjpy, 60))
factors['dxy_beta_60'] = build('dxy_beta_60', lambda s, p: rbeta(p, dxy, 60))
factors['eurusd_beta_60'] = build('eurusd_beta_60', lambda s, p: rbeta(p, eurusd, 60))
factors['wti_beta_60'] = build('wti_beta_60', lambda s, p: rbeta(p, px['WTI'], 60))
factors['xau_cop_beta_60'] = build('xau_cop_beta_60', lambda s, p: rbeta(p, px['XAU'], 60) - rbeta(p, px['COPPER'], 60))

# --- Family B: conditional beta (stress resilience) ---
vix_up = ret1(vix) > 0.004
factors['vix_up_ret_resilience_60'] = build('vix_up_ret_resilience_60',
    lambda s, p: ret1(p).where(vix_up.reindex(p.index).fillna(False)).rolling(60, min_periods=12).mean().reindex(INDEX))
dxy_up = ret1(dxy) > 0.005
factors['dxy_up_resilience_60'] = build('dxy_up_resilience_60',
    lambda s, p: ret1(p).where(dxy_up.reindex(p.index).fillna(False)).rolling(60, min_periods=12).mean().reindex(INDEX))
jpy_up = ret1(usdjpy) > 0.004
factors['usdjpy_up_beta_60'] = build('usdjpy_up_beta_60',
    lambda s, p: rbeta(p, usdjpy, 60, cond=jpy_up))
factors['vix_beta_cond_60x20'] = build('vix_beta_cond_60x20',
    lambda s, p: rbeta(p, vix, 60, cond=(vix.pct_change().rolling(20).sum() > 0).reindex(p.index)))
factors['yield_beta_cond_60x20'] = build('yield_beta_cond_60x20',
    lambda s, p: rbeta(p, us10, 60, cond=(us10.pct_change().rolling(20).sum() > 0).reindex(p.index)))

# --- Family C: volatility / risk shape ---
factors['vol_z_60'] = build('vol_z_60', lambda s, p: (rstd(p, 60) - rstd(p, 60).rolling(252, min_periods=60).mean()) / rstd(p, 60).rolling(252, min_periods=60).std())
factors['maxdd_60'] = build('maxdd_60', lambda s, p: -(vseries(p).rolling(60, min_periods=30).max() / vseries(p) - 1.0).reindex(INDEX))
factors['tail_ratio_20'] = build('tail_ratio_20', lambda s, p: (vseries(p).rolling(20, min_periods=10).quantile(0.95) - vseries(p)) / (vseries(p) - vseries(p).rolling(20, min_periods=10).quantile(0.05)))
factors['vol_skew_20'] = build('vol_skew_20', lambda s, p: (vseries(p).rolling(20, min_periods=10).quantile(0.9) - vseries(p).rolling(20, min_periods=10).median()) / rstd(p, 20))
factors['ret_kurt_30'] = build('ret_kurt_30', lambda s, p: vseries(p).pct_change().rolling(30, min_periods=15).kurt().reindex(INDEX))

# --- Family D: trend / momentum variants ---
factors['tw_mom_40'] = build('tw_mom_40', lambda s, p: (vseries(p) / vseries(p).shift(40) - 1.0).reindex(INDEX))
factors['range_pos_20'] = build('range_pos_20', lambda s, p: (vseries(p) - vseries(p).rolling(20, min_periods=10).min()) / (vseries(p).rolling(20, min_periods=10).max() - vseries(p).rolling(20, min_periods=10).min()))
factors['rel_mom_20'] = build('rel_mom_20', lambda s, p: retk(p, 20) - retk(px.mean(axis=1), 20).reindex(p.index))
factors['streak_20'] = build('streak_20', lambda s, p: (ret1(p) > 0).astype(int).rolling(20, min_periods=5).mean().reindex(INDEX))

def fwd_panel(h):
    out = {}
    for s in WATCH:
        v = vseries(px[s])
        out[s] = (v.shift(-h) / v - 1.0).reindex(INDEX)
    return pd.DataFrame(out).sort_index()

# library panels for rho
lib_panels = {'usdcny_beta_60': factors['usdcny_beta_60']}

res = {}
for name, f in factors.items():
    fz = f.replace([np.inf, -np.inf], np.nan)
    icd = cross_sectional_ic(fz, fwd_panel(H))
    st = ic_stats(icd)
    ranks = fz.rank(axis=1)
    to = ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1)
    cov = fz.notna().sum().sum() / (len(fz) * len(WATCH))
    rhos = {}
    for lname, lp in lib_panels.items():
        rhos[lname] = round(spearman_panel_rho(fz, lp.reindex(fz.index)), 4) if len(fz) else None
    maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
    reg = regime_split(icd)
    decay = {}
    for hh in [1, 2, 3, 5, 10, 20]:
        icd_h = cross_sectional_ic(fz, fwd_panel(hh))
        decay[hh] = round(ic_stats(icd_h)['ic'], 4)
    gate = (abs(st['ic']) >= 0.0070) and (abs(st['icir']) >= 0.0840)
    res[name] = {'ic': st['ic'], 'icir': st['icir'], 'hit': st['hit'], 'n_dates': st['n_dates'],
                 'avg_assets': st['avg_n'], 'turnover_10d': to, 'coverage': cov,
                 'rho_lib': rhos, 'max_lib_rho': maxrho, 'regime': reg, 'decay': decay,
                 'flag': 'PASS' if gate else 'fail'}
    print(f"== {name} == {'PASS' if gate else 'FAIL'}  ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_a={st['avg_n']:.1f}")
    print(f"   to10={to:.3f} cov={cov:.3f} maxrho={maxrho:.3f} rhos={rhos}")
    print(f"   regime={ {k: [round(x,4) for x in v] for k,v in reg.items()} }")
    print(f"   decay={decay}")

json.dump(res, open('scripts/_miner3_0224_screen_results.json', 'w'), indent=1, default=str)
print("\nSaved scripts/_miner3_0224_screen_results.json")
