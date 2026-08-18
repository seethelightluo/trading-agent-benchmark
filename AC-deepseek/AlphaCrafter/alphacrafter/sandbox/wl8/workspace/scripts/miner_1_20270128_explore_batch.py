"""miner_1 cycle 2027-01-28: explore fresh cross-asset candidates on the 15-asset universe.
Data visible through 2027-01-27 (previous completed trading day). No future leakage.

Admission gates (shared, 15-asset universe): |IC_10d| >= 0.0070 and |ICIR_10d| >= 0.0840.
Audit: max_abs_library_correlation vs usdcny_beta_60 (decoded artifact) + recomputed ensemble panels.

Candidates (volatility/rotation/curvature family, distinct mechanics from library):
  V1 vol_trend_20           : 20d RV / 60d RV (rising-vol underperformance)
  V2 cv_ratio_20            : 20d return / 20d RV (risk-adjusted momentum)
  V3 curv_20x60             : (20d mom) - (60d mom), scaled by 60d trend sign (trend curvature)
  V4 cn_resid_vol_20        : asset RV / 000300.SH RV (index-relative vol)
  V5 xau_up_resilience_60   : mean asset return on XAU-up days (safe-haven rotation)
  V6 spx_dd_resilience_60   : mean asset return when SPX in 20d drawdown (defensive quality)
  V7 crypto_lead_mom_20     : BTC 20d mom * 60d corr(asset, BTC) (crypto-led risk-on)
  V8 downskew_20            : 20d downside vol / 20d total vol (skew/vol asymmetry)
  V9 usdcny_rv_ratio_20     : USDCNY 20d RV / asset 20d RV (CNY stress premium)
  V10 vol_up_ret_60         : mean asset return on VIX-up days (hedge value)
Also drift re-validation of effective library factors through 2027-01-27.
"""
import sys, os, io, json, zlib, base64
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner_3_20261203_common import (WATCH, load_prices, load_macro, zscore_series,
                                     cross_sectional_ic, ic_stats, regime_split,
                                     spearman_panel_rho)

ASOF = '2027-01-27'
H = 10
IC_THR, ICIR_THR = 0.0070, 0.0840

px = load_prices(ASOF)
macro = load_macro(ASOF)
fwd = px.shift(-H) / px - 1.0
vix, dxy, usdcny = macro['VIX'], macro['DXY'], macro['USDCNY']
spx, xau, btc = px['SPX'], px['XAU'], px['BTC']
cn = px['000300.SH']
us10 = px['US10Y']

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


def rolling_beta(y, x, w, minp=None):
    vy, vx = y.dropna(), x.dropna()
    df = pd.concat([vy.rename('y'), vx.rename('x')], axis=1, sort=True).dropna()
    if minp is None:
        minp = max(6, int(w * 0.4))
    cov = df['y'].rolling(w, min_periods=minp).cov(df['x'])
    var = df['x'].rolling(w, min_periods=minp).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan).reindex(px.index)


def rcorr(y, x, w, minp=None):
    vy, vx = y.dropna(), x.dropna()
    df = pd.concat([vy.rename('a'), vx.rename('b')], axis=1, sort=True).dropna()
    if minp is None:
        minp = max(4, int(w * 0.5))
    return df['a'].rolling(w, min_periods=minp).corr(df['b']).reindex(px.index)


def build_factor(name, fn):
    cols = {}
    for s in WATCH:
        try:
            cols[s] = fn(s, px[s])
        except Exception:
            cols[s] = np.nan
    return pd.DataFrame(cols).sort_index()


usdcny_rv20 = rstd(usdcny.pct_change(), 20)
usdcny_rv60 = rstd(usdcny.pct_change(), 60)
btc_mom20 = retk(btc, 20)
btc_r = btc.pct_change()
xau_up = xau.pct_change() > 0.002
spx_r = spx.pct_change()
spx_dd = spx / spx.rolling(20, min_periods=10).max() - 1.0
vix_up = vix.pct_change() > 0.01
cn_rv20 = rstd(cn.pct_change(), 20)
mom20 = retk(px, 20)
mom60 = retk(px, 60)


def v1(s, p):
    return rstd(p.pct_change(), 20) / rstd(p.pct_change(), 60).replace(0, np.nan)


def v2(s, p):
    rv = rstd(p.pct_change(), 20)
    return retk(p, 20) / rv.replace(0, np.nan)


def v3(s, p):
    g = np.sign(mom60[s].reindex(p.index))
    return (retk(p, 20) - retk(p, 60).reindex(p.index)) * g


def v4(s, p):
    return rstd(p.pct_change(), 20) / cn_rv20.reindex(p.index).replace(0, np.nan)


def v5(s, p):
    r = p.pct_change()
    m = xau_up.reindex(p.index).fillna(False)
    return r.where(m).rolling(60, min_periods=12).mean().reindex(px.index)


def v6(s, p):
    r = p.pct_change()
    m = (spx_dd < 0).reindex(p.index).fillna(False)
    return r.where(m).rolling(60, min_periods=12).mean().reindex(px.index)


def v7(s, p):
    return btc_mom20.reindex(p.index) * rcorr(p.pct_change(), btc_r, 60)


def v8(s, p):
    r = p.pct_change()
    dn = r.where(r < 0)
    dn_rv = dn.rolling(20, min_periods=10).std()
    tot_rv = r.rolling(20, min_periods=10).std()
    return (dn_rv / tot_rv.replace(0, np.nan)).reindex(px.index)


def v9(s, p):
    return usdcny_rv20.reindex(p.index) / rstd(p.pct_change(), 20).replace(0, np.nan)


def v10(s, p):
    r = p.pct_change()
    m = vix_up.reindex(p.index).fillna(False)
    return r.where(m).rolling(60, min_periods=12).mean().reindex(px.index)


factors = {
    'vol_trend_20': build_factor('vol_trend_20', v1),
    'cv_ratio_20': build_factor('cv_ratio_20', v2),
    'curv_20x60': build_factor('curv_20x60', v3),
    'cn_resid_vol_20': build_factor('cn_resid_vol_20', v4),
    'xau_up_resilience_60': build_factor('xau_up_resilience_60', v5),
    'spx_dd_resilience_60': build_factor('spx_dd_resilience_60', v6),
    'crypto_lead_mom_20': build_factor('crypto_lead_mom_20', v7),
    'downskew_20': build_factor('downskew_20', v8),
    'usdcny_rv_ratio_20': build_factor('usdcny_rv_ratio_20', v9),
    'vol_up_ret_60': build_factor('vol_up_ret_60', v10),
}

# ---------------- library panels (fresh recompute) ----------------
def decode_usdcny():
    d = json.load(open('factors/usdcny_beta_60.json'))
    art = d['validation']['signal_artifact']['data']
    df = pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(art)).decode()))
    df.index = pd.to_datetime(df[df.columns[0]])
    return df.drop(columns=df.columns[0])


lib_panels = {'usdcny_beta_60': decode_usdcny()}
lib_panels['mom_10d_skip5'] = px / px.shift(15) - 1.0

ycond20 = (us10 / us10.shift(20) - 1.0) > 0
vb = {}
for s in WATCH:
    vb[s] = rolling_beta(px[s].pct_change(), vix.pct_change(), 60)
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame(vb).sort_index()

yb = {}
for s in WATCH:
    yb[s] = rolling_beta(px[s].pct_change(), us10.pct_change(), 60)
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
        gate = (abs(st['ic']) >= IC_THR) and (abs(st['icir']) >= ICIR_THR) and len(icd) > 0
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

json.dump(res, open('scripts/_miner1_20270128_screen_results.json', 'w'), indent=1, default=str, allow_nan=True)

# ---------------- drift re-validation of effective library ----------------
print("\n=== Drift re-validation of library factors through 2027-01-27 ===")
combo = {'mom_10d_skip5': px / px.shift(15) - 1.0,
         'vix_beta_cond_60x20': pd.DataFrame(vb).sort_index(),
         'yield_beta_cond_60x20': pd.DataFrame(yb).sort_index()}
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

uz = decode_usdcny().apply(zscore_series, axis=0).reindex(fwd.index)
st_u = ic_stats(cross_sectional_ic(uz, fwd, min_assets=8))
print(f"usdcny_beta_60 (artifact): full ic={st_u['ic']:.4f} icir={st_u['icir']:.4f} n={st_u['n_dates']}")

# ---------------- live snapshot of best candidates ----------------
print("\n=== Live snapshot of strongest candidates on 2027-01-27 (z-scored) ===")
order = sorted(res, key=lambda k: -(abs(res[k].get('ic', 0)) if isinstance(res[k].get('ic'), float) else 0))
for name in order[:4]:
    try:
        fz = factors[name].apply(zscore_series, axis=0)
        last = fz.iloc[-1].dropna().sort_values(ascending=False)
        print(f"{name}:")
        print(last.round(2).to_string())
    except Exception as e:
        print(name, 'ERR', e)
print("\nSaved -> scripts/_miner1_20270128_screen_results.json")