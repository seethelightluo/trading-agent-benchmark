"""miner_1 exploration 2027-06-03 (ASOF 2027-06-02): regime-aware factor screen.
Regime: whipsaw momentum, JPY at 168 (carry tail risk), VIX pinned 9.0, 4 frozen feeds
(000688.SH, CN10Y, NDX, SOX). Focus: vol-scaled trend, trend efficiency, FX/carry beta,
drawdown/range persistence, up/down capture. IC evaluated BOTH on all-15 and live-only
cross-section (frozen assets contribute artificial constants -> live-only is primary).
Gates: |IC|>=0.0070, |ICIR|>=0.0840 at H=10.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, zlib, base64, io
from miner_3_20261203_common import WATCH, load_prices, load_macro, regime_split, spearman_panel_rho

ASOF = '2027-06-02'
H = 10
FROZEN = ['000688.SH', 'CN10Y', 'NDX', 'SOX']
LIVE = [s for s in WATCH if s not in FROZEN]

px = load_prices(ASOF)
macro = load_macro(ASOF)
INDEX = px.index

def vseries(s): return s.dropna()
def ret1(s): return vseries(s).pct_change().reindex(INDEX)
def retk(s, k): 
    v = vseries(s)
    return (v / v.shift(k) - 1.0).reindex(INDEX)
def rstd(s, w, minp=None):
    v = vseries(s)
    if minp is None: minp = max(3, int(w * 0.5))
    return v.rolling(w, min_periods=minp).std().reindex(INDEX)
def rbeta(y, x, w, cond=None, minp=None):
    vy, vx = vseries(y), vseries(x)
    df = pd.concat([vy.rename('y'), vx.rename('x')], axis=1, sort=True).dropna()
    if cond is not None:
        c = cond.reindex(df.index).fillna(False).astype(bool)
    else:
        c = pd.Series(True, index=df.index)
    ym, xm = df['y'].where(c), df['x'].where(c)
    if minp is None: minp = max(6, int(w * 0.4))
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

us10 = px['US10Y']; vix = macro['VIX']; dxy = macro['DXY']
usdjpy = macro['USDJPY']; usdcny = macro['USDCNY']
btc = px['BTC']; eth = px['ETH']; xau = px['XAU']; wti = px['WTI']

factors = {}

# --- vol-scaled trend (whipsaw-regime stabilizers) ---
factors['sharpe_mom_20_60'] = build('sharpe_mom_20_60', lambda s, p: retk(p, 20) / rstd(p, 60))
factors['sharpe_mom_10_30'] = build('sharpe_mom_10_30', lambda s, p: retk(p, 10) / rstd(p, 30))
factors['sharpe_mom_40_120'] = build('sharpe_mom_40_120', lambda s, p: retk(p, 40) / rstd(p, 120))

# --- trend efficiency (Kaufman ER) ---
def kaufman_er(s, p, w=20):
    v = vseries(p)
    net = (v - v.shift(w)).abs()
    path = v.diff().abs().rolling(w).sum()
    return (net / path).reindex(INDEX)
factors['kaufman_er_20'] = build('kaufman_er_20', lambda s, p: kaufman_er(s, p, 20))
factors['kaufman_er_40'] = build('kaufman_er_40', lambda s, p: kaufman_er(s, p, 40))

# --- vol regime z-score (vol-of-vol timing) ---
def vol_z(s, p, w=20, long=120):
    r = ret1(p)
    rv = r.rolling(w, min_periods=10).std()
    base = rv.rolling(long, min_periods=60).median()
    sd = rv.rolling(long, min_periods=60).std()
    z = (rv - base) / sd.replace(0, np.nan)
    return z.reindex(INDEX)
factors['vol_z_20_120'] = build('vol_z_20_120', lambda s, p: vol_z(s, p, 20, 120))

# --- FX/carry beta: USDJPY (JPY at 168, carry unwind tail risk) ---
factors['usdjpy_beta_60'] = build('usdjpy_beta_60', lambda s, p: rbeta(p, usdjpy, 60))
factors['usdjpy_beta_20'] = build('usdjpy_beta_20', lambda s, p: rbeta(p, usdjpy, 20))

# --- DXY beta (safe-haven flow) ---
factors['dxy_beta_60'] = build('dxy_beta_60', lambda s, p: rbeta(p, dxy, 60))

# --- drawdown from high / range position (trend persistence) ---
def dd_from_high(s, p, w=120):
    v = vseries(p)
    return (v / v.rolling(w, min_periods=60).max() - 1.0).reindex(INDEX)
factors['dd_from_high_120'] = build('dd_from_high_120', lambda s, p: dd_from_high(s, p, 120))

def hi_lo_pos(s, p, w=60):
    v = vseries(p)
    hi = v.rolling(w, min_periods=30).max(); lo = v.rolling(w, min_periods=30).min()
    rng = (hi - lo).replace(0, np.nan)
    return ((v - lo) / rng).reindex(INDEX)
factors['hi_lo_pos_60'] = build('hi_lo_pos_60', lambda s, p: hi_lo_pos(s, p, 60))

# --- up/down capture (asymmetry) ---
def updown_capture(s, p, w=60):
    r = ret1(p)
    up = r.where(r > 0, 0.0).rolling(w, min_periods=20).mean()
    dn = r.where(r < 0, 0.0).rolling(w, min_periods=20).mean()
    return ((up + dn.abs()) * np.sign(up + dn.abs())).reindex(INDEX)  # net capture magnitude
factors['updown_capture_60'] = build('updown_capture_60', lambda s, p: updown_capture(s, p, 60))

# --- crypto cluster beta (BTC-driven risk appetite) ---
factors['btc_beta_20'] = build('btc_beta_20', lambda s, p: rbeta(p, btc, 20))
factors['btc_beta_60'] = build('btc_beta_60', lambda s, p: rbeta(p, btc, 60))

# --- VIX-up resilience (defensive, short-vix-beta analog) ---
vix_up = ret1(vix) > 0.005
factors['vix_up_res_60'] = build('vix_up_res_60',
    lambda s, p: ret1(p).where(vix_up.reindex(p.index).fillna(False)).rolling(60, min_periods=12).mean().reindex(INDEX))

# --- yield-rising beta conditional ---
ychg20 = (us10 / us10.shift(20) - 1.0) > 0
factors['us10y_up_beta_60'] = build('us10y_up_beta_60',
    lambda s, p: rbeta(p, us10, 60, cond=ychg20))

# --- relative momentum vs cross-sectional median (breadth-robust) ---
def rel_mom(s, p, w=20):
    v = vseries(p)
    mom = (v / v.shift(w) - 1.0)
    med = pd.concat([(v2 / v2.shift(w) - 1.0) for v2 in px[s2].dropna() for s2 in WATCH if s2 == s], axis=1)
    return mom.reindex(INDEX)
# simpler: cross-sectional median momentum
mom_panel = retk(px, 20)
med_mom = mom_panel.median(axis=1)
factors['rel_mom_20'] = pd.DataFrame({s: (mom_panel[s] - med_mom) for s in WATCH}).sort_index()

# --- commodity-gold rotation ---
factors['xau_wti_ratio_mom_20'] = pd.DataFrame({s: retk(xau if s in ('XAU','WTI','COPPER') else px[s], 20) * np.sign(retk(xau,20) - retk(wti,20)).reindex(INDEX) for s in WATCH}).sort_index()

# ---------- library panels ----------
def decode_usdcny():
    d = json.load(open('factors/usdcny_beta_60.json'))
    art = d['validation']['signal_artifact']['data']
    csv_txt = zlib.decompress(base64.b64decode(art)).decode()
    df = pd.read_csv(io.StringIO(csv_txt))
    return df.set_index(df.columns[0])
lib_panels = {'usdcny_beta_60': decode_usdcny()}
lib_panels['mom_10d_skip5'] = retk(px, 15)
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame({s: rbeta(px[s], vix, 60, cond=(vix.pct_change().rolling(20).sum() > 0)) for s in WATCH}).sort_index()
lib_panels['yield_beta_cond_60x20'] = pd.DataFrame({s: rbeta(px[s], us10, 60, cond=ychg20) for s in WATCH}).sort_index()

# ---------- evaluation ----------
def cross_sectional_ic(fpanel, fwd_panel, assets=None, min_assets=8):
    recs = []
    common = fpanel.index.intersection(fwd_panel.index)
    for d in common:
        f = fpanel.loc[d]; r = fwd_panel.loc[d]
        if assets is not None:
            m = f.notna() & r.notna() & np.isfinite(f) & np.isfinite(r) & pd.Series(True, index=f.index).reindex(assets).fillna(False).values
            m = pd.Series(m, index=f.index)
        else:
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

def fwd_panel(h):
    out = {}
    for s in WATCH:
        v = vseries(px[s])
        out[s] = (v.shift(-h) / v - 1.0).reindex(INDEX)
    return pd.DataFrame(out).sort_index()

fwd10 = fwd_panel(H)
res = {}
for name, f in factors.items():
    fz = f.replace([np.inf, -np.inf], np.nan)
    icd_all = cross_sectional_ic(fz, fwd10)
    icd_live = cross_sectional_ic(fz, fwd10, assets=LIVE)
    st_all = ic_stats(icd_all); st_live = ic_stats(icd_live)
    ranks = fz.rank(axis=1)
    to = ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1)
    cov = fz.notna().sum().sum() / (len(fz) * len(WATCH))
    rhos = {}
    for lname, lp in lib_panels.items():
        rhos[lname] = round(spearman_panel_rho(fz, lp.reindex(fz.index)), 4) if len(fz) else None
    maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
    reg = regime_split(icd_live if len(icd_live) >= len(icd_all)//2 else icd_all)
    decay = {}
    for hh in [1, 2, 3, 5, 10, 20]:
        icd_h = cross_sectional_ic(fz, fwd_panel(hh), assets=LIVE)
        decay[hh] = round(ic_stats(icd_h)['ic'], 4)
    gate = (abs(st_live['ic']) >= 0.0070) and (abs(st_live['icir']) >= 0.0840)
    res[name] = {'ic_live': st_live['ic'], 'icir_live': st_live['icir'], 'hit_live': st_live['hit'],
                 'n_live': st_live['n_dates'], 'avg_a_live': st_live['avg_n'],
                 'ic_all': st_all['ic'], 'icir_all': st_all['icir'], 'n_all': st_all['n_dates'],
                 'turnover_10d': to, 'coverage': cov, 'rho_lib': rhos, 'max_lib_rho': maxrho,
                 'regime': reg, 'decay': decay, 'flag': 'PASS' if gate else 'fail'}
    print(f"== {name} == {'PASS' if gate else 'FAIL'}  ic_live={st_live['ic']:.4f} icir_live={st_live['icir']:.4f} hit={st_live['hit']:.3f} n_live={st_live['n_dates']} avg_a={st_live['avg_n']:.1f} | ic_all={st_all['ic']:.4f} n_all={st_all['n_dates']}")
    print(f"   to10={to:.3f} cov={cov:.3f} maxrho={maxrho:.3f} rhos={rhos}")
    print(f"   regime={ {k: [round(x,4) for x in v] for k,v in reg.items()} }")
    print(f"   decay={decay}")

json.dump(res, open('scripts/_miner1_0603_explore_results.json', 'w'), indent=1, default=str)
print("\nSaved scripts/_miner1_0603_explore_results.json")
