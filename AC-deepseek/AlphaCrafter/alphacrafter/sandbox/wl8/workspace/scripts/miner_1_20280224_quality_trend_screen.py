"""miner_1 cycle 2028-02-24 (ASOF 2028-02-23): quality-trend / volatility-adjusted momentum screen.
Gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 at H=10 on 15-asset cross-section (>=8 valid/date).
Also revalidates usdcny_beta_60 (only persisted active library factor).
Rho computed vs recomputed library (usdcny_beta_60) + informational rho vs ensemble factors.
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
print(f"universe: {len(WATCH)} assets, price rows {len(px)}, date range {px.index[0].date()} .. {px.index[-1].date()}")

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

# ------------------------------------------------------------------
# macro series
vix = macro['VIX']
us10 = px['US10Y']
usdcny = macro['USDCNY']

# ------------------------------------------------------------------
# Library revalidation
factors = {}
factors['usdcny_beta_60'] = build('usdcny_beta_60', lambda s, p: rbeta(p, usdcny, 60))

# --- Family A: volatility-adjusted / quality momentum ---
# A1: Sharpe-style momentum: 40d return scaled by 40d daily vol (per-period units)
factors['mom_sharpe_40'] = build('mom_sharpe_40', lambda s, p: retk(p, 40) / rstd(p, 40))
# A2: efficiency ratio (Kaufman): |net 40d move| / sum(|daily moves| 40d)
factors['eff_ratio_40'] = build('eff_ratio_40', lambda s, p: (vseries(p) / vseries(p).shift(40) - 1.0).abs() / vseries(p).pct_change().abs().rolling(40, min_periods=15).sum()).reindex(INDEX) if False else None
def eff_ratio(p, w):
    v = vseries(p)
    net = (v / v.shift(w) - 1.0).abs()
    path = v.pct_change().abs().rolling(w, min_periods=int(w * 0.4)).sum()
    return (net / path).reindex(INDEX)
factors['eff_ratio_20'] = build('eff_ratio_20', lambda s, p: eff_ratio(p, 20))
factors['eff_ratio_40'] = build('eff_ratio_40', lambda s, p: eff_ratio(p, 40))
# A3: up-day / down-day return asymmetry over 40d (positive = persistent buyers)
def updown_ratio(p, w):
    r = ret1(p)
    up = r.where(r > 0).rolling(w, min_periods=int(w * 0.4)).mean()
    dn = r.where(r < 0).rolling(w, min_periods=int(w * 0.4)).mean()
    return (up / dn.abs()).reindex(INDEX)
factors['updown_ratio_40'] = build('updown_ratio_40', lambda s, p: updown_ratio(p, 40))
# A4: streak persistence: fraction of up days in last 40d x |net move| direction
factors['streak_mom_40'] = build('streak_mom_40', lambda s, p: ((ret1(p) > 0).astype(float).rolling(40, min_periods=15).mean() * 2 - 1).reindex(INDEX))
# A5: close position within 40d range (0=bottom,1=top) - range momentum
factors['range_pos_40'] = build('range_pos_40', lambda s, p: ((vseries(p) - vseries(p).rolling(40, min_periods=15).min()) / (vseries(p).rolling(40, min_periods=15).max() - vseries(p).rolling(40, min_periods=15).min())).reindex(INDEX))
# A6: 20d momentum vs 120d momentum (trend acceleration)
factors['mom_accel_20x120'] = build('mom_accel_20x120', lambda s, p: retk(p, 20) - retk(p, 120))
# A7: vol-scaled 10d mom (faster variant)
factors['mom_sharpe_10'] = build('mom_sharpe_10', lambda s, p: retk(p, 10) / rstd(p, 10))

def fwd_panel(h):
    out = {}
    for s in WATCH:
        v = vseries(px[s])
        out[s] = (v.shift(-h) / v - 1.0).reindex(INDEX)
    return pd.DataFrame(out).sort_index()

# library panels for rho: active lib factor + ensemble factors (informational)
lib_panels = {'usdcny_beta_60': factors['usdcny_beta_60']}
# recompute ensemble members for informational correlation
vix_up20 = (vix.pct_change().rolling(20).sum() > 0).reindex(INDEX)
us10_up20 = (us10.pct_change().rolling(20).sum() > 0).reindex(INDEX)
ens = {}
ens['mom_10d_skip5'] = build('mom_10d_skip5', lambda s, p: retk(p, 10).shift(5))
ens['vix_beta_cond_60x20'] = build('vix_beta_cond_60x20', lambda s, p: rbeta(p, vix, 60, cond=vix_up20))
ens['yield_beta_cond_60x20'] = build('yield_beta_cond_60x20', lambda s, p: rbeta(p, us10, 60, cond=us10_up20))

res = {}
allfac = dict(factors)
allfac.update(ens)
for name, f in allfac.items():
    fz = f.replace([np.inf, -np.inf], np.nan)
    icd = cross_sectional_ic(fz, fwd_panel(H))
    st = ic_stats(icd)
    ranks = fz.rank(axis=1)
    to = ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1)
    cov = fz.notna().sum().sum() / (len(fz) * len(WATCH))
    rhos = {}
    for lname, lp in lib_panels.items():
        rhos[lname] = round(spearman_panel_rho(fz, lp.reindex(fz.index)), 4) if len(fz) else None
    ens_rho = {}
    for lname, lp in ens.items():
        if lname != name:
            ens_rho[lname] = round(spearman_panel_rho(fz, lp.reindex(fz.index)), 4) if len(fz) else None
    maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
    reg = regime_split(icd)
    decay = {}
    for hh in [1, 2, 3, 5, 10, 20]:
        icd_h = cross_sectional_ic(fz, fwd_panel(hh))
        decay[hh] = round(ic_stats(icd_h)['ic'], 4)
    gate = (abs(st['ic']) >= 0.0070) and (abs(st['icir']) >= 0.0840)
    res[name] = {'ic': st['ic'], 'icir': st['icir'], 'hit': st['hit'], 'n_dates': st['n_dates'],
                 'avg_assets': st['avg_n'], 'turnover_10d': to, 'coverage': cov,
                 'rho_lib': rhos, 'rho_ens': ens_rho, 'max_lib_rho': maxrho,
                 'regime': reg, 'decay': decay, 'flag': 'PASS' if gate else 'fail'}
    print(f"== {name} == {'PASS' if gate else 'FAIL'}  ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_a={st['avg_n']:.1f}")
    print(f"   to10={to:.3f} cov={cov:.3f} max_lib_rho={maxrho:.3f} rho_lib={rhos} rho_ens={ens_rho}")
    print(f"   regime={ {k: [round(x,4) for x in v] for k,v in reg.items()} }")
    print(f"   decay={decay}")

json.dump(res, open('scripts/_miner1_0224_screen_results.json', 'w'), indent=1, default=str)
print("\nSaved scripts/_miner1_0224_screen_results.json")