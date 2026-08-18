"""miner_3 cycle 2028-03-23 (ASOF 2028-03-22): risk-off defensive-structure factor screen.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 at H=10 on the 15-asset cross-section (>=8 valid/date).
Library panels for rho: usdcny_beta_60 (active), mom_10d_skip5 / vix_beta_cond_60x20 / yield_beta_cond_60x20 (fallback).
NOTE: 000688/CN10Y/NDX/SOX frozen in worldline (flat closes) -> degenerate rolling stats handled via dropna; coverage reported.
Focus: defensive resilience / asym participation / trend-persistence variants with low library overlap.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json
from miner_3_20261203_common import WATCH, load_prices, load_macro, spearman_panel_rho, regime_split

ASOF = '2028-03-22'
H = 10
RECENT_W = 252  # ~12m timeliness window

px = load_prices(ASOF)
macro = load_macro(ASOF)
INDEX = px.index
print(f"price panel: {px.shape[0]} dates x {px.shape[1]} assets; macro {macro.shape}; asof={ASOF}")

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

def rcorr(a, b, w, minp=None):
    va, vb = vseries(a), vseries(b)
    df = pd.concat([va.rename('a'), vb.rename('b')], axis=1, sort=True).dropna()
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

# --- macro / cross-asset references ---
vix = macro['VIX']
dxy = macro['DXY']
usdcny = macro['USDCNY']
us10 = px['US10Y']
cn10y = px['CN10Y']
xau = px['XAU']

# equal-weight cross-asset market return panel
mkt_ret = ret1(px.mean(axis=1))
# safe-haven basket: XAU + CN10Y (+US10Y as separate variant)
haven_ret = ret1(pd.concat([xau, cn10y], axis=1).mean(axis=1))

factors = {}

# --- Library revalidation (rho reference + timeliness) ---
factors['usdcny_beta_60'] = build('usdcny_beta_60', lambda s, p: rbeta(p, usdcny, 60))
factors['mom_10d_skip5'] = build('mom_10d_skip5', lambda s, p: retk(p, 15) - retk(p, 5))
factors['vix_beta_cond_60x20'] = build('vix_beta_cond_60x20',
    lambda s, p: rbeta(p, vix, 60, cond=(vix.pct_change().rolling(20).sum() > 0).reindex(p.index)))
factors['yield_beta_cond_60x20'] = build('yield_beta_cond_60x20',
    lambda s, p: rbeta(p, us10, 60, cond=(us10.pct_change().rolling(20).sum() > 0).reindex(p.index)))

# --- Family A: defensive / risk-off structure ---
# A1 gold correlation (haven adjacency)
factors['gold_corr_60'] = build('gold_corr_60', lambda s, p: rcorr(p, xau, 60))
# A2 beta to safe-haven basket XAU+CN10Y
factors['haven_beta_60'] = build('haven_beta_60', lambda s, p: rbeta(p, haven_ret, 60))
# A3 up/down capture asymmetry vs cross-asset market (up-capture minus down-capture)
def updown_capture(s, p, w=60, minp=20):
    r = ret1(p)
    m = mkt_ret.reindex(r.index)
    df = pd.concat([r.rename('r'), m.rename('m')], axis=1).dropna()
    up = df['r'].where(df['m'] > 0)
    dn = df['r'].where(df['m'] < 0)
    uc = up.rolling(w, min_periods=int(w*0.4)).mean()
    dc = dn.rolling(w, min_periods=int(w*0.4)).mean()
    return (uc - dc).reindex(INDEX)
factors['updown_capture_60'] = build('updown_capture_60', updown_capture)
# A4 drawdown recovery score: how far recovered from 60d low within 60d range, momentum-weighted
def recovery_60(s, p, w=60, minp=30):
    v = vseries(p)
    lo = v.rolling(w, min_periods=minp).min()
    hi = v.rolling(w, min_periods=minp).max()
    pos = (v - lo) / (hi - lo).replace(0, np.nan)
    mom5 = retk(p, 5)
    return (pos + 0.5 * np.sign(mom5) * pos.abs() ** 0.5).reindex(INDEX)
factors['recovery_60'] = build('recovery_60', recovery_60)

# --- Family B: trend persistence / risk-scaled momentum ---
# B1 3-month trend minus 1-month (medium-term trend with pullback penalty)
factors['xtrend_40x10'] = build('xtrend_40x10', lambda s, p: retk(p, 40) - retk(p, 10))
# B2 risk-adjusted 20d momentum (t-stat like)
factors['vol_adj_mom_20x60'] = build('vol_adj_mom_20x60',
    lambda s, p: retk(p, 20).reindex(INDEX) / rstd(p, 60).reindex(INDEX))
# B3 flip-consistent momentum: 20d mom gated by 5d direction agreement
def flip_mom(s, p, kw=20, ks=5):
    m20 = retk(p, kw)
    m5 = retk(p, ks)
    gate = np.sign(m5)
    return (m20 * gate).reindex(INDEX)
factors['flip_mom_20x5'] = build('flip_mom_20x5', flip_mom)

# --- Family C: EM / macro conditional beta asymmetry ---
# C1 beta to USDCNY on CNY-weakness (EM stress) days only
cny_up_days = (ret1(usdcny) > 0.001).reindex(INDEX)
factors['usdcny_up_beta_60'] = build('usdcny_up_beta_60',
    lambda s, p: rbeta(p, usdcny, 60, cond=cny_up_days))
# C2 VIX beta asymmetry: beta on VIX-rise days minus beta on VIX-fall days
vix_up = (ret1(vix) > 0).reindex(INDEX)
vix_dn = (ret1(vix) < 0).reindex(INDEX)
def vix_asym(s, p, w=60):
    b_up = rbeta(p, vix, w, cond=vix_up)
    b_dn = rbeta(p, vix, w, cond=vix_dn)
    return (b_up - b_dn).reindex(INDEX)
factors['vix_beta_asym_60'] = build('vix_beta_asym_60', vix_asym)

# --- forward returns ---
def fwd_panel(h):
    out = {}
    for s in WATCH:
        v = vseries(px[s])
        out[s] = (v.shift(-h) / v - 1.0).reindex(INDEX)
    return pd.DataFrame(out).sort_index()

lib_panels = {
    'usdcny_beta_60': factors['usdcny_beta_60'],
    'mom_10d_skip5': factors['mom_10d_skip5'],
    'vix_beta_cond_60x20': factors['vix_beta_cond_60x20'],
    'yield_beta_cond_60x20': factors['yield_beta_cond_60x20'],
}

def fast_ic(fdf, rdf, min_assets=8):
    mask = fdf.notna() & rdf.notna() & np.isfinite(fdf) & np.isfinite(rdf)
    n = mask.sum(axis=1)
    f = fdf.where(mask).rank(axis=1)
    r = rdf.where(mask).rank(axis=1)
    fm = f.sub(f.mean(axis=1), axis=0)
    rm = r.sub(r.mean(axis=1), axis=0)
    num = (fm * rm).sum(axis=1)
    den = np.sqrt((fm ** 2).sum(axis=1) * (rm ** 2).sum(axis=1))
    ic = (num / den.replace(0, np.nan)).where(n >= min_assets)
    return ic.dropna()

def ic_stats_fast(ic):
    ic = ic.astype(float)
    return {'ic': float(ic.mean()), 'icir': float(ic.mean() / ic.std(ddof=1)) if ic.std(ddof=1) else 0.0,
            'hit': float((ic > 0).mean()), 'n_dates': int(len(ic))}

res = {}
for name, f in factors.items():
    fz = f.replace([np.inf, -np.inf], np.nan)
    fwd10 = fwd_panel(H)
    icd = fast_ic(fz, fwd10)
    st = ic_stats_fast(icd)
    # recent-window timeliness
    if len(icd):
        icd_r = icd[icd.index >= icd.index[-1] - pd.Timedelta(days=RECENT_W)]
        st_r = ic_stats_fast(icd_r) if len(icd_r) else {'ic': np.nan, 'icir': np.nan, 'hit': np.nan, 'n_dates': 0}
    else:
        st_r = {'ic': np.nan, 'icir': np.nan, 'hit': np.nan, 'n_dates': 0}
    ranks = fz.rank(axis=1)
    to = ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1)
    cov = fz.notna().sum().sum() / (len(fz) * len(WATCH))
    rhos = {}
    for lname, lp in lib_panels.items():
        rhos[lname] = round(spearman_panel_rho(fz, lp.reindex(fz.index)), 4) if len(fz) else None
    maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
    decay = {}
    for hh in [1, 2, 3, 5, 10, 20]:
        icd_h = fast_ic(fz, fwd_panel(hh))
        decay[hh] = round(float(icd_h.mean()) if len(icd_h) else np.nan, 4)
    reg = {}
    for lab, m in [('2020-2021', icd.index < pd.Timestamp('2022-01-01')),
                   ('2022-2023', (icd.index >= pd.Timestamp('2022-01-01')) & (icd.index < pd.Timestamp('2024-01-01'))),
                   ('2024+', icd.index >= pd.Timestamp('2024-01-01'))]:
        sub = icd[m]
        if len(sub):
            reg[lab] = [round(float(sub.mean()), 4), round(float(sub.mean() / sub.std(ddof=1)), 4) if sub.std(ddof=1) else 0.0, int(len(sub))]
    gate = (abs(st['ic']) >= 0.0070) and (abs(st['icir']) >= 0.0840)
    res[name] = {'ic': st['ic'], 'icir': st['icir'], 'hit': st['hit'], 'n_dates': st['n_dates'],
                 'ic_recent252': st_r['ic'], 'icir_recent252': st_r['icir'], 'n_recent': st_r['n_dates'],
                 'turnover_10d': to, 'coverage': cov,
                 'rho_lib': rhos, 'max_lib_rho': maxrho, 'decay': decay, 'regime': reg,
                 'flag': 'PASS' if gate else 'fail'}
    print(f"== {name} == {'PASS' if gate else 'FAIL'}  ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']}")
    print(f"   recent252 ic={st_r['ic']:.4f} icir={st_r['icir']:.4f} n={st_r['n_dates']} | to10={to:.3f} cov={cov:.3f} maxrho={maxrho:.3f}")
    print(f"   rhos={rhos}")
    print(f"   decay={decay}")
    print(f"   regime={ {k: [round(x,4) if isinstance(x,float) else x for x in v] for k,v in reg.items()} }")

json.dump(res, open('scripts/_miner3_20280323_screen_results.json', 'w'), indent=1, default=str)
print("\nSaved scripts/_miner3_20280323_screen_results.json")