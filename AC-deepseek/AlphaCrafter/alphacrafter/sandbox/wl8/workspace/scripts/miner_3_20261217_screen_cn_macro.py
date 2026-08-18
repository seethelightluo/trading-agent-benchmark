"""miner_3 cycle 2026-12-17: screen CN-anchored + macro-beta candidates.
Admission gates: |IC| >= 0.0070, |ICIR| >= 0.0840 at 10d horizon, on 15-asset cross-section.
Library correlation computed vs usdcny_beta_60 (decoded real artifact) + recomputed ensemble factors.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd, json, zlib, base64, io
from miner_3_20261203_common import WATCH, MACRO, load_prices, load_macro, cross_sectional_ic, ic_stats, regime_split, spearman_panel_rho, zscore_series

ASOF = '2026-12-16'
H = 10  # admission horizon

px = load_prices(ASOF)
macro = load_macro(ASOF)
cn = px['000300.SH']
fwd = px.shift(-H) / px - 1.0

# ---------- factor builders ----------
def ret(s, k):
    return s / s.shift(k) - 1.0

def rolling_beta(y, x, w, cond=None):
    """rolling beta of y on x over window w; cond: boolean array to restrict days (None = all)."""
    yv = y.to_frame('y'); xv = x.to_frame('x')
    df = yv.join(xv)
    if cond is not None:
        df['c'] = cond.reindex(df.index).fillna(False).astype(bool)
    else:
        df['c'] = True
    out = pd.Series(np.nan, index=df.index)
    # vectorized rolling cov/var on masked series
    mask = df['c']
    ym = df['y'].where(mask); xm = df['x'].where(mask)
    cov = ym.rolling(w, min_periods=int(w*0.6)).cov(xm)
    var = xm.rolling(w, min_periods=int(w*0.6)).var()
    out = cov / var
    return out.replace([np.inf, -np.inf], np.nan)

def rolling_corr(y, x, w):
    return y.rolling(w, min_periods=int(w*0.6)).corr(x)

def build_factor(name, fn):
    cols = {}
    for s in WATCH:
        try:
            cols[s] = fn(s, px[s])
        except Exception as e:
            cols[s] = np.nan
    f = pd.DataFrame(cols).sort_index()
    return f

factors = {}

# F1: csi300_hedged_mom_60 — residual momentum vs CN index (asset 60d skip5 mom minus beta*CN mom)
def f1(s, p):
    id_mom = p / p.shift(65) - 1.0
    cn_mom = (cn / cn.shift(65) - 1.0).reindex(p.index)
    b = rolling_beta(p.pct_change(), cn.pct_change(), 60)
    return id_mom - b * cn_mom
factors['csi300_hedged_mom_60'] = build_factor('csi300_hedged_mom_60', f1)

# F2: csi300_vol_ratio_20 — asset 20d RV / CN 20d RV (expect negative IC)
def f2(s, p):
    rv_a = p.pct_change().rolling(20).std()
    rv_cn = cn.pct_change().rolling(20).std().reindex(p.index)
    return rv_a / rv_cn
factors['csi300_vol_ratio_20'] = build_factor('csi300_vol_ratio_20', f2)

# F3: cn_corr_down_20 = -corr(asset, CN, 20d) (expect positive)
def f3(s, p):
    return -rolling_corr(p.pct_change(), cn.pct_change(), 20)
factors['cn_corr_down_20'] = build_factor('cn_corr_down_20', f3)

# F4: csi300_down_beta_60 — beta on days CN fell >1% (expect negative)
def f4(s, p):
    cnr = cn.pct_change()
    cond = (cnr < -0.01)
    return rolling_beta(p.pct_change(), cnr, 60, cond=cond)
factors['csi300_down_beta_60'] = build_factor('csi300_down_beta_60', f4)

# F5: cn_rate_cond_mom_60 — asset mom_60_skip5 * sign(CN10Y 120d change); cut-cycle momentum
cn10y = px['CN10Y']
cn10y_chg120 = (cn10y / cn10y.shift(120) - 1.0)
def f5(s, p):
    m = p / p.shift(65) - 1.0
    sig = np.sign(cn10y_chg120.reindex(p.index))
    return m * sig
factors['cn_rate_cond_mom_60'] = build_factor('cn_rate_cond_mom_60', f5)

# F6: usdjpy_beta_60 — beta vs USDJPY (carry/weak-yen proxy), expect + for carry assets
usdjpy = macro['USDJPY']
def f6(s, p):
    return rolling_beta(p.pct_change(), usdjpy.pct_change(), 60)
factors['usdjpy_beta_60'] = build_factor('usdjpy_beta_60', f6)

# F7: vix_up_resilience_60 — mean asset daily ret on days VIX rose >2% over last 60d (expect +)
vix = macro['VIX']
vix_up = vix.pct_change() > 0.02
def f7(s, p):
    r = p.pct_change()
    m = vix_up.reindex(p.index).fillna(False)
    out = r.where(m).rolling(60, min_periods=12).mean()
    return out
factors['vix_up_resilience_60'] = build_factor('vix_up_resilience_60', f7)

# F8: cn_bear_rel_strength_20 — asset 20d ret if CN 20d ret < -5%, else 0 (expect +)
cn_bear = (cn / cn.shift(20) - 1.0) < -0.05
def f8(s, p):
    r20 = p / p.shift(20) - 1.0
    return r20.where(cn_bear.reindex(p.index), 0.0)
factors['cn_bear_rel_strength_20'] = build_factor('cn_bear_rel_strength_20', f8)

# ---------- library panels ----------
def decode_usdcny():
    d = json.load(open('factors/usdcny_beta_60.json'))
    art = d['validation']['signal_artifact']['data']
    csv_txt = zlib.decompress(base64.b64decode(art)).decode()
    df = pd.read_csv(io.StringIO(csv_txt))
    return df.set_index(df.columns[0])

lib_panels = {'usdcny_beta_60': decode_usdcny()}

def mom_10d_skip5():
    return px / px.shift(15) - 1.0   # close_{t-5}/close_{t-15} - 1
lib_panels['mom_10d_skip5'] = mom_10d_skip5()

def vix_beta_cond_60x20():
    # beta vs VIX over 60d, conditional on VIX 20d change positive
    vchg20 = (vix / vix.shift(20) - 1.0) > 0
    out = {}
    for s in WATCH:
        r = px[s].pct_change()
        cond = vchg20.reindex(px.index).fillna(False)
        cov = r.where(cond).rolling(60, min_periods=24).cov(vix.pct_change().where(cond))
        var = vix.pct_change().where(cond).rolling(60, min_periods=24).var()
        out[s] = cov / var
    return pd.DataFrame(out).sort_index().replace([np.inf, -np.inf], np.nan)
lib_panels['vix_beta_cond_60x20'] = vix_beta_cond_60x20()

def yield_beta_cond_60x20():
    # beta vs US10Y over 60d, conditional on US10Y 20d change positive
    us10 = px['US10Y']
    ychg20 = (us10 / us10.shift(20) - 1.0) > 0
    out = {}
    for s in WATCH:
        r = px[s].pct_change()
        cond = ychg20.reindex(px.index).fillna(False)
        cov = r.where(cond).rolling(60, min_periods=24).cov(us10.pct_change().where(cond))
        var = us10.pct_change().where(cond).rolling(60, min_periods=24).var()
        out[s] = cov / var
    return pd.DataFrame(out).sort_index().replace([np.inf, -np.inf], np.nan)
lib_panels['yield_beta_cond_60x20'] = yield_beta_cond_60x20()

# ---------- evaluate ----------
res = {}
for name, f in factors.items():
    fz = f.apply(zscore_series, axis=0)
    icd = cross_sectional_ic(fz, fwd, min_assets=8)
    st = ic_stats(icd)
    # turnover: mean abs rank change at 10d spacing
    ranks = fz.rank(axis=1)
    to = ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1)
    n_valid = fz.notna().sum().sum()
    cov = n_valid / (len(fz) * len(WATCH))
    # library correlation
    rhos = {}
    for lname, lp in lib_panels.items():
        lpc = lp.reindex(fz.index)
        if isinstance(lpc, pd.Series):
            lpc = lpc.to_frame()
        rhos[lname] = spearman_panel_rho(fz, lpc)
    maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
    reg = regime_split(icd)
    decay = {}
    for hh in [1, 2, 3, 5, 10, 20]:
        fh = px.shift(-hh) / px - 1.0
        icd_h = cross_sectional_ic(fz, fh, min_assets=8)
        decay[hh] = float(ic_stats(icd_h)['ic']) if len(icd_h) else np.nan
    gate = (abs(st['ic']) >= 0.0070) and (abs(st['icir']) >= 0.0840)
    res[name] = {'ic': st['ic'], 'icir': st['icir'], 'hit': st['hit'], 'n_dates': st['n_dates'],
                 'avg_assets': st['avg_n'], 'turnover_10d': to, 'coverage': cov,
                 'rho_lib': rhos, 'max_lib_rho': maxrho, 'regime': reg, 'decay': decay, 'flag': 'PASS' if gate else 'fail'}
    print(f"== {name} == {'PASS' if gate else 'FAIL'}")
    print(f"   ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n={st['n_dates']} avg_assets={st['avg_n']:.1f}")
    print(f"   turnover10d={to:.3f} coverage={cov:.3f} max_lib_rho={maxrho:.3f} rhos={ {k: (round(v,3) if v==v else None) for k,v in rhos.items()} }")
    print(f"   regime={ {k: [round(x,4) for x in v] for k,v in reg.items()} }")
    print(f"   decay={ {k: round(v,4) if v==v else None for k,v in decay.items()} }")

json.dump(res, open('scripts/_miner3_cycle23_screen_results.json', 'w'), indent=1, default=str)
print("\nSaved -> scripts/_miner3_cycle23_screen_results.json")