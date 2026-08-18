"""miner_3 cycle 2026-12-31: robust CN-anchored + macro-beta factor screen (v2, bugfixed).
Fixes: per-asset valid-trading-day series for returns/rolling stats; forward returns per asset
over h valid days; ic_stats always returns avg_n; robust prints.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 at H=10 on 15-asset cross-section (>=8 valid assets/date).
Library rho vs usdcny_beta_60 (decoded real artifact) + robust-recomputed ensemble factors.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, zlib, base64, io
from miner_3_20261203_common import WATCH, load_prices, load_macro, regime_split

ASOF = '2026-12-30'
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

cn = px['000300.SH']
cn10y = px['CN10Y']
vix = macro['VIX']
usdjpy = macro['USDJPY'] if 'USDJPY' in macro.columns else None

factors = {}

cn_mom60 = retk(cn, 65)
def f1(s, p):
    return retk(p, 65) - rbeta(p, cn, 60) * cn_mom60
factors['csi300_hedged_mom_60'] = build('csi300_hedged_mom_60', f1)

cn_rv20 = rstd(cn, 20)
def f2(s, p):
    return rstd(p, 20) / cn_rv20
factors['csi300_vol_ratio_20'] = build('csi300_vol_ratio_20', f2)

def f3(s, p):
    return -rcorr(p, cn, 20)
factors['cn_corr_down_20'] = build('cn_corr_down_20', f3)

cn_r1 = ret1(cn)
cn_down_cond = (cn_r1 < -0.01)
def f4(s, p):
    return rbeta(p, cn, 60, cond=cn_down_cond)
factors['csi300_down_beta_60'] = build('csi300_down_beta_60', f4)

cn10y_chg120 = retk(cn10y, 120)
def f5(s, p):
    return retk(p, 65) * np.sign(cn10y_chg120.reindex(p.index))
factors['cn_rate_cond_mom_60'] = build('cn_rate_cond_mom_60', f5)

def f6(s, p):
    return rbeta(p, usdjpy, 60)
factors['usdjpy_beta_60'] = build('usdjpy_beta_60', f6)

vix_up = vix.pct_change() > 0.02
def f7(s, p):
    r = ret1(p)
    m = vix_up.reindex(p.index).fillna(False)
    return r.where(m).rolling(60, min_periods=12).mean().reindex(INDEX)
factors['vix_up_resilience_60'] = build('vix_up_resilience_60', f7)

cn_bear = retk(cn, 20) < -0.05
def f8(s, p):
    return retk(p, 20).where(cn_bear.reindex(p.index), 0.0)
factors['cn_bear_rel_strength_20'] = build('cn_bear_rel_strength_20', f8)

def f9(s, p):
    r = ret1(p)
    m = cn_down_cond.reindex(p.index).fillna(False)
    down_rv = r.where(m).rolling(60, min_periods=10).std()
    all_rv = r.rolling(60, min_periods=30).std()
    return (down_rv / all_rv).reindex(INDEX)
factors['csi300_down_vol_60'] = build('csi300_down_vol_60', f9)

cn10y_chg40 = retk(cn10y, 40)
def f10(s, p):
    return retk(p, 45) * np.sign(cn10y_chg40.reindex(p.index))
factors['cn10y_led_mom_40'] = build('cn10y_led_mom_40', f10)

# ---------- library panels ----------
def decode_usdcny():
    d = json.load(open('factors/usdcny_beta_60.json'))
    art = d['validation']['signal_artifact']['data']
    csv_txt = zlib.decompress(base64.b64decode(art)).decode()
    df = pd.read_csv(io.StringIO(csv_txt))
    return df.set_index(df.columns[0])

lib_panels = {'usdcny_beta_60': decode_usdcny(), 'mom_10d_skip5': retk(px, 15)}
vchg20 = (vix / vix.shift(20) - 1.0) > 0
us10 = px['US10Y']
ychg20 = (us10 / us10.shift(20) - 1.0) > 0
vb = {}; yb = {}
for s in WATCH:
    vb[s] = rbeta(px[s], vix, 60, cond=vchg20)
    yb[s] = rbeta(px[s], us10, 60, cond=ychg20)
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame(vb).sort_index()
lib_panels['yield_beta_cond_60x20'] = pd.DataFrame(yb).sort_index()

# ---------- evaluation ----------
def cross_sectional_ic(fpanel, fwd_panel, min_assets=8):
    recs = []
    common = fpanel.index.intersection(fwd_panel.index)
    for d in common:
        f = fpanel.loc[d]; r = fwd_panel.loc[d]
        m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r)
        n = int(m.sum())
        if n >= min_assets:
            ic = f[m].corr(r[m], method='spearman')
            if not np.isnan(ic):
                recs.append((d, ic, n))
    return pd.DataFrame(recs, columns=['date', 'ic', 'n']).set_index('date')

def ic_stats(icdf):
    if len(icdf) == 0:
        return {'ic': np.nan, 'icir': np.nan, 'hit': np.nan, 'n_dates': 0, 'avg_n': 0}
    ic = icdf['ic'].mean()
    sd = icdf['ic'].std(ddof=1)
    icir = ic / sd if sd and not np.isnan(sd) and sd > 0 else 0.0
    return {'ic': ic, 'icir': icir, 'hit': (icdf['ic'] > 0).mean(), 'n_dates': len(icdf), 'avg_n': icdf['n'].mean()}

def spearman_panel_rho(a, b):
    common = a.index.intersection(b.index)
    rhos = []
    for d in common:
        av = a.loc[d]; bv = b.loc[d]
        m = av.notna() & bv.notna() & np.isfinite(av) & np.isfinite(bv)
        if m.sum() >= 6:
            r = av[m].corr(bv[m], method='spearman')
            if not np.isnan(r):
                rhos.append(r)
    return float(np.mean(rhos)) if rhos else np.nan

def fwd_panel(h):
    out = {}
    for s in WATCH:
        v = vseries(px[s])
        out[s] = (v.shift(-h) / v - 1.0).reindex(INDEX)
    return pd.DataFrame(out).sort_index()

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

json.dump(res, open('scripts/_miner3_cycle24_screen_results.json', 'w'), indent=1, default=str)
print("\nSaved scripts/_miner3_cycle24_screen_results.json")