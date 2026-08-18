"""miner_3 cycle 2027-04-22 (ASOF 2027-04-21): rate/FX macro + dispersion screen.
Gates: |IC| >= 0.0070, |ICIR| >= 0.0840 at H=10 on 15-asset cross-section (>=8 valid/date).
Library rho vs usdcny_beta_60 (decoded artifact) + ensemble factors (robust recompute).
NOTE: NDX/SOX/000688/CN10Y frozen -> use live cross-section; report frozen set.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, zlib, base64, io
from miner_3_20261203_common import WATCH, load_prices, load_macro, regime_split

ASOF = '2027-04-21'
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
usdcny = macro['USDCNY'] if 'USDCNY' in macro.columns else None
eurusd = macro['EURUSD'] if 'EURUSD' in macro.columns else None

factors = {}

# 1. USDJPY beta outright (carry/FX-beta). JPY strengthening regime active.
factors['usdjpy_beta_60'] = build('usdjpy_beta_60', lambda s, p: rbeta(p, usdjpy, 60))
factors['usdjpy_beta_40'] = build('usdjpy_beta_40', lambda s, p: rbeta(p, usdjpy, 40))

# 2. USDJPY rising-day beta conditional (carry stress resilience)
jpy_up = ret1(usdjpy) > 0.004
factors['usdjpy_up_beta_60'] = build('usdjpy_up_beta_60',
    lambda s, p: rbeta(p, usdjpy, 60, cond=jpy_up))

# 3. DXY rising-day avg return resilience (safe-haven)
dxy_up = ret1(dxy) > 0.005
factors['dxy_up_resilience_60'] = build('dxy_up_resilience_60',
    lambda s, p: ret1(p).where(dxy_up.reindex(p.index).fillna(False)).rolling(60, min_periods=12).mean().reindex(INDEX))

# 4. DXY omega (strong/weak excursion ratio over 60d): high DXY-beta assets
dxy_ret = ret1(dxy)
factors['dxy_beta_60'] = build('dxy_beta_60', lambda s, p: rbeta(p, dxy, 60))

# 5. Rate vol uncertainty sensitivity: 60d asset vol corr to US10Y vol
rv60 = rstd(us10, 60)
factors['rate_vol_corr_120'] = build('rate_vol_corr_120',
    lambda s, p: rcorr(rstd(p, 60), rv60, 120, minp=40))

# 6. US10Y rising-rate conditional beta (risk-off tie to rates)
ychg20 = (us10 / us10.shift(20) - 1.0) > 0
factors['us_rate_beta_cond_60x20'] = build('us_rate_beta_cond_60x20',
    lambda s, p: rbeta(p, us10, 60, cond=(us10.pct_change().rolling(20).sum() > 0).reindex(p.index)))

# 7. EURUSD beta (risk-on proxy, EUR strength = risk appetite)
factors['eurusd_beta_60'] = build('eurusd_beta_60', lambda s, p: rbeta(p, eurusd, 60))

# 8. USDCNY rising-day beta conditional
cny_up = ret1(usdcny) > 0.003
factors['usdcny_up_beta_60'] = build('usdcny_up_beta_60',
    lambda s, p: rbeta(p, usdcny, 60, cond=cny_up))

# 9. Dispersion: asset 60d vol / US10Y 60d vol z-score (defensive tilt to rate vol)
def f_vol_ratio_us(s, p):
    r = ret1(p)
    rv = r.rolling(60, min_periods=30).std()
    us_rv = us10.pct_change().rolling(60, min_periods=30).std()
    ratio = (rv / us_rv).replace([np.inf, -np.inf], np.nan)
    med = ratio.median(); mad = (ratio - med).abs().median()
    if mad == 0 or np.isnan(mad):
        return ratio * 0.0
    return ((ratio - med) / (1.4826 * mad)).clip(-5, 5)
factors['vol_ratio_us_rate_z_60'] = build('vol_ratio_us_rate_z_60', f_vol_ratio_us)

# 10. WTI-led momentum breakout: 20d momentum * sign of commodity cycle
wti_chg60 = retk(px['WTI'], 60)
factors['wti_led_mom_20'] = build('wti_led_mom_20',
    lambda s, p: retk(p, 20) * np.sign(wti_chg60.reindex(p.index)))

# 11. Crypto-XAU alternating strength: risk-on/risk-off rotation proxy
xau_chg20 = retk(px['XAU'], 20)
factors['xau_led_mom_20'] = build('xau_led_mom_20',
    lambda s, p: retk(p, 20) * np.sign(xau_chg20.reindex(p.index)))

# 12. 20d raw momentum (short-term continuation on live cross-section)
factors['mom_20d'] = build('mom_20d', lambda s, p: retk(p, 20))

# 13. VIX-level sensitivity: high-beta-to-VIX short via vol-adjust (defensive)
vchg20 = (vix / vix.shift(20) - 1.0) > 0
factors['vix_beta_cond_60x20'] = build('vix_beta_cond_60x20',
    lambda s, p: rbeta(p, vix, 60, cond=vchg20))

# ---------- library panels ----------
def decode_usdcny():
    d = json.load(open('factors/usdcny_beta_60.json'))
    art = d['validation']['signal_artifact']['data']
    csv_txt = zlib.decompress(base64.b64decode(art)).decode()
    df = pd.read_csv(io.StringIO(csv_txt))
    return df.set_index(df.columns[0])
lib_panels = {'usdcny_beta_60': decode_usdcny()}
lib_panels['mom_10d_skip5'] = retk(px, 15)
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

json.dump(res, open('scripts/_miner3_0422_screen_results.json', 'w'), indent=1, default=str)
print("\nSaved scripts/_miner3_0422_screen_results.json")
