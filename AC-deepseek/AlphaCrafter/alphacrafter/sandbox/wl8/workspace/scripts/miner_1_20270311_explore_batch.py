"""miner_1 cycle 2027-03-11: explore fresh cross-asset candidates on the 15-asset universe.
Data visible through 2027-03-10 (previous completed trading day). No future leakage.

Admission gates (shared, 15-asset universe): |IC_10d| >= 0.0070 and |ICIR_10d| >= 0.0840.
Audit: max_abs_library_correlation vs usdcny_beta_60 (decoded artifact) + recomputed fallback
ensemble panels (mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20).

Candidates (new angle this cycle - regime-conditional cross-asset beta / quality momentum / FX-led):
  A1 trend_eff_60        : Kaufman efficiency ratio 60d (trend quality, not plain momentum)
  A2 mom_rv_ratio_20     : 20d momentum scaled by 60d RV (risk-adjusted momentum)
  A3 yield_level_beta_60 : 60d beta to US10Y gated by rate LEVEL above median (level regime, not drift)
  A4 xau_stress_beta_60  : 60d beta to XAU conditional on VIX-up days (flight-to-safety sensitivity)
  A5 copper_trend_beta_60: 60d beta to COPPER conditional on copper 40d uptrend (cyclical exposure)
  A6 us10y_duration_60   : -60d corr(asset ret, US10Y ret) (rate-sensitivity/duration proxy)
  A7 hsi_led_mom_40      : sign(HSI 40d mom) * 40d asset mom (China complex regime alignment)
  A8 jump_reversal_5x60  : -(5d ret)/RV60 (short-term reversal after jumps, vol-normalized)
  A9 drawdown_depth_60   : (close - 60d max)/60d max (underwater depth, mean-reversion pull)
  A10 rv_regime_rank_60  : cross-sectional rank of RV60 (defensive tilt when own vol high)
Also drift re-validation of active library factors (usdcny_beta_60 artifact + fallback trio).
"""
import sys, os, io, json, zlib, base64
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner_3_20261203_common import (WATCH, load_prices, load_macro, zscore_series,
                                     cross_sectional_ic, ic_stats, regime_split,
                                     spearman_panel_rho)

ASOF = '2027-03-10'
H = 10
IC_THR, ICIR_THR = 0.0070, 0.0840

px = load_prices(ASOF)
macro = load_macro(ASOF)
fwd = px.shift(-H) / px - 1.0
vix, dxy, usdcny = macro['VIX'], macro['DXY'], macro['USDCNY']
spx, xau, btc, wti, us10, hsi, copper = px['SPX'], px['XAU'], px['BTC'], px['WTI'], px['US10Y'], px['HSI'], px['COPPER']

print(f"Universe: {len(WATCH)} assets, price dates {px.index[0].date()}..{px.index[-1].date()} ({len(px)} rows)")
print(f"Admission gates: |IC|>={IC_THR}, |ICIR|>={ICIR_THR}, horizon {H}d, min_assets>=8\n")

def retk(s, k):
    v = s.dropna()
    return (v / v.shift(k) - 1.0).reindex(px.index)

def rstd(s, w, minp=None):
    v = s.dropna()
    if minp is None:
        minp = max(3, int(w * 0.5))
    return v.rolling(w, min_periods=minp).std().reindex(px.index)

def rolling_beta(y, x, w, minp=None, cond=None):
    vy, vx = y.dropna(), x.dropna()
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
    return (cov / var).replace([np.inf, -np.inf], np.nan).reindex(px.index)

def rcorr(y, x, w, minp=None, cond=None):
    vy, vx = y.dropna(), x.dropna()
    df = pd.concat([vy.rename('a'), vx.rename('b')], axis=1, sort=True).dropna()
    if cond is not None:
        c = cond.reindex(df.index).fillna(False).astype(bool)
    else:
        c = pd.Series(True, index=df.index)
    ma, mb = df['a'].where(c), df['b'].where(c)
    if minp is None:
        minp = max(4, int(w * 0.5))
    return ma.rolling(w, min_periods=minp).corr(mb).reindex(px.index)

def rkaufman_er(s, w, minp=None):
    v = s.dropna()
    if minp is None:
        minp = max(10, int(w * 0.5))
    chg = (v - v.shift(w)).abs()
    path = v.diff().abs().rolling(w, min_periods=minp).sum()
    return (chg / path.replace(0, np.nan)).reindex(px.index)

def build_factor(name, fn):
    cols = {}
    for s in WATCH:
        try:
            cols[s] = fn(s, px[s])
        except Exception:
            cols[s] = np.nan
    return pd.DataFrame(cols).sort_index()

spx_r, xau_r = spx.pct_change(), xau.pct_change()
wti_r, us10_r = wti.pct_change(), us10.pct_change()
hsi_r, copper_r = hsi.pct_change(), copper.pct_change()
vix_r = vix.pct_change()
rv60 = px.apply(lambda s: rstd(s, 60))
rv20 = px.apply(lambda s: rstd(s, 20))

# A1 Kaufman efficiency 60d
def a1(s, p):
    return rkaufman_er(p, 60)
# A2 20d momentum / 60d RV
def a2(s, p):
    m = retk(p, 20)
    return (m / rv60[s].reindex(p.index)).reindex(p.index)
# A3 beta to US10Y gated by rate level above rolling median (persistently high-rate regime)
rate_level_hi = (us10 > us10.rolling(250, min_periods=60).median()).reindex(px.index)
def a3(s, p):
    return rolling_beta(p.pct_change(), us10_r, 60, cond=rate_level_hi)
# A4 beta to XAU on VIX-up (stress) days - flight to safety sensitivity
vix_up = (vix_r > 0).reindex(px.index)
def a4(s, p):
    return rolling_beta(p.pct_change(), xau_r, 60, cond=vix_up)
# A5 beta to COPPER when copper 40d trend up (cyclical up-tilt)
copper_up40 = (copper / copper.shift(40) - 1.0) > 0
def a5(s, p):
    return rolling_beta(p.pct_change(), copper_r, 60, cond=copper_up40.reindex(px.index))
# A6 duration-like US10Y sensitivity (negative corr with rate returns)
def a6(s, p):
    return (-1.0 * rcorr(p.pct_change(), us10_r, 60)).reindex(p.index)
# A7 HSI-led momentum for China complex
hsi_mom40 = retk(hsi, 40)
def a7(s, p):
    return np.sign(hsi_mom40.reindex(p.index).fillna(0)) * retk(p, 40).reindex(p.index)
# A8 jump reversal: -(5d return)/RV60
def a8(s, p):
    m5 = retk(p, 5)
    return (-m5 / rv60[s].reindex(p.index)).reindex(p.index)
# A9 drawdown depth vs 60d max
def a9(s, p):
    hi = p.rolling(60, min_periods=20).max()
    return ((p - hi) / hi).reindex(p.index)
# A10 cross-sectional rank z of RV60 (defensive tilt when own vol high)
def a10(s, p):
    v = rv60[s].reindex(p.index)
    return v.rank(axis=0) / v.notna().sum() - 0.5 if False else v

factors = {
    'trend_eff_60': build_factor('trend_eff_60', a1),
    'mom_rv_ratio_20': build_factor('mom_rv_ratio_20', a2),
    'yield_level_beta_60': build_factor('yield_level_beta_60', a3),
    'xau_stress_beta_60': build_factor('xau_stress_beta_60', a4),
    'copper_trend_beta_60': build_factor('copper_trend_beta_60', a5),
    'us10y_duration_60': build_factor('us10y_duration_60', a6),
    'hsi_led_mom_40': build_factor('hsi_led_mom_40', a7),
    'jump_reversal_5x60': build_factor('jump_reversal_5x60', a8),
    'drawdown_depth_60': build_factor('drawdown_depth_60', a9),
}
# A10: cross-sectional rank of RV60 within each date
rvr = rv60.rank(axis=1, pct=True) - 0.5
factors['rv_regime_rank_60'] = rvr

# ---------------- library panels (fresh recompute) ----------------
def decode_artifact(fn):
    d = json.load(open(fn))
    art = d['validation']['signal_artifact']['data']
    df = pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(art)).decode()))
    df.index = pd.to_datetime(df[df.columns[0]])
    return df.drop(columns=df.columns[0])

lib_panels = {'usdcny_beta_60': decode_artifact('factors/usdcny_beta_60.json')}
lib_panels['mom_10d_skip5'] = px / px.shift(15) - 1.0
vb = {s: rolling_beta(px[s].pct_change(), vix_r, 60, cond=(vix_r > 0)) for s in WATCH}
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame(vb).sort_index()
yb = {s: rolling_beta(px[s].pct_change(), us10_r, 60, cond=(us10 / us10.shift(20) - 1.0 > 0)) for s in WATCH}
lib_panels['yield_beta_cond_60x20'] = pd.DataFrame(yb).sort_index()

# ---------------- evaluate candidates ----------------
res = {}
for name, f in factors.items():
    try:
        fz = f.apply(zscore_series, axis=0)
        icd = cross_sectional_ic(fz, fwd, min_assets=8)
        st = ic_stats(icd)
        ranks = fz.rank(axis=1)
        to = ranks.diff(10).abs().mean().mean() / (len(WATCH) - 1)
        n_valid = int(fz.notna().sum().sum())
        cov = n_valid / (len(fz) * len(WATCH))
        rhos = {}
        for lname, lp in lib_panels.items():
            lpc = lp.reindex(fz.index)
            rhos[lname] = spearman_panel_rho(fz, lpc)
        maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
        reg = regime_split(icd)
        decay = {}
        for hh in [1, 2, 3, 5, 10, 20]:
            fh = px.shift(-hh) / px - 1.0
            icd_h = cross_sectional_ic(fz, fh, min_assets=8)
            decay[hh] = float(ic_stats(icd_h)['ic']) if len(icd_h) else np.nan
        gate = (abs(st['ic']) >= IC_THR) and (abs(st['icir']) >= ICIR_THR) and st['n_dates'] >= 60
        res[name] = {'ic': st['ic'], 'icir': st['icir'], 'hit': st['hit'], 'n_dates': st['n_dates'],
                     'avg_assets': st.get('avg_n', np.nan), 'turnover_10d': to, 'coverage': cov,
                     'rho_lib': rhos, 'max_lib_rho': maxrho, 'regime': reg, 'decay': decay,
                     'flag': 'PASS' if gate else 'fail'}
        print(f"== {name} == {'PASS' if gate else 'fail'}")
        print(f"   ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n_dates={st['n_dates']} "
              f"avg_assets={st.get('avg_n', float('nan')):.1f}")
        print(f"   turnover10d={to:.3f} coverage={cov:.3f} max_lib_rho={maxrho:.3f} "
              f"rhos={ {k: (round(v, 3) if v == v else None) for k, v in rhos.items()} }")
        print(f"   regime={ {k: [round(x, 4) for x in v] for k, v in reg.items()} }")
        print(f"   decay={ {k: round(v, 4) if v == v else None for k, v in decay.items()} }")
    except Exception as e:
        res[name] = {'error': str(e), 'flag': 'err'}
        print(f"== {name} == ERROR: {e}")

json.dump(res, open('scripts/_miner1_20270311_screen_results.json', 'w'), indent=1, default=str, allow_nan=True)

# ---------------- drift re-validation of active library ----------------
print("\n=== Drift re-validation of active library + fallback ensemble through 2027-03-10 ===")
combo = {'mom_10d_skip5': px / px.shift(15) - 1.0,
         'vix_beta_cond_60x20': lib_panels['vix_beta_cond_60x20'],
         'yield_beta_cond_60x20': lib_panels['yield_beta_cond_60x20']}
for name, f in combo.items():
    fz = f.apply(zscore_series, axis=0)
    full = ic_stats(cross_sectional_ic(fz, fwd, min_assets=8))
    sl = fz.iloc[-90:]; fl = fwd.reindex(sl.index)
    last_90 = ic_stats(cross_sectional_ic(sl, fl, min_assets=8))
    s2 = fz.iloc[-180:]; f2 = fwd.reindex(s2.index)
    last_180 = ic_stats(cross_sectional_ic(s2, f2, min_assets=8))
    print(f"{name}: full ic={full['ic']:.4f} icir={full['icir']:.4f} n={full['n_dates']} | "
          f"90d ic={last_90['ic']:.4f} icir={last_90['icir']:.4f} n={last_90['n_dates']} | "
          f"180d ic={last_180['ic']:.4f} icir={last_180['icir']:.4f} n={last_180['n_dates']}")

uz = decode_artifact('factors/usdcny_beta_60.json').apply(zscore_series, axis=0).reindex(fwd.index)
st_u = ic_stats(cross_sectional_ic(uz, fwd, min_assets=8))
zu = uz.iloc[-90:]; flu = fwd.reindex(zu.index)
st_u90 = ic_stats(cross_sectional_ic(zu, flu, min_assets=8))
su = uz.iloc[-180:]; fsu = fwd.reindex(su.index)
st_u180 = ic_stats(cross_sectional_ic(su, fsu, min_assets=8))
print(f"usdcny_beta_60 (artifact): full ic={st_u['ic']:.4f} icir={st_u['icir']:.4f} n={st_u['n_dates']} | "
      f"90d ic={st_u90['ic']:.4f} icir={st_u90['icir']:.4f} n={st_u90['n_dates']} | "
      f"180d ic={st_u180['ic']:.4f} icir={st_u180['icir']:.4f} n={st_u180['n_dates']}")

# ---------------- live snapshot of strongest candidates ----------------
print("\n=== Live snapshot of strongest candidates on 2027-03-10 (z-scored) ===")
order = sorted(res, key=lambda k: -(abs(res[k].get('ic', 0)) if isinstance(res[k].get('ic'), float) else 0))
for name in order[:4]:
    try:
        fz = factors[name].apply(zscore_series, axis=0)
        last = fz.iloc[-1].dropna().sort_values(ascending=False)
        print(f"{name}:")
        print(last.round(2).to_string())
    except Exception as e:
        print(name, 'ERR', e)
print("\nSaved -> scripts/_miner1_20270311_screen_results.json")