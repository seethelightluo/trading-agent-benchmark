"""miner_1 cycle 2027-02-11: explore fresh cross-asset candidates on the 15-asset universe.
Data visible through 2027-02-10 (previous completed trading day). No future leakage.

Admission gates (shared, 15-asset universe): |IC_10d| >= 0.0070 and |ICIR_10d| >= 0.0840.
Audit: max_abs_library_correlation vs usdcny_beta_60 (decoded artifact) + recomputed fallback
ensemble panels (mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20).

Candidates (vol term-structure / correlation dynamics / rotation / skew family):
  W1 vol_convexity_20x60 : (RV20 - RV60)/RV60  -- short-horizon vol premium (vol TS steepness)
  W2 mom_corr_spx_60     : 20d change in 60d rolling corr(asset, SPX) -- correlation momentum
  W3 xau_diverg_20       : XAU 20d return - asset 20d return (safe-haven rotation spread)
  W4 btc_trend_corr_60   : sign(BTC 60d mom) * 60d corr(asset, BTC) (crypto regime alignment)
  W5 ret_skew_60         : 60d skewness of daily returns (lottery/left-tail exposure)
  W6 volume_z_20         : 20d volume z-score (attention/liquidity pulse; 9 assets have volume)
  W7 wti_down_beta_60    : 60d beta of asset to WTI returns on WTI-down days (energy stress)
  W8 drawup_60           : (close - 60d min)/(60d max - 60d min) -- recovery strength position
Also drift re-validation of effective library factor usdcny_beta_60 and fallback trio.
"""
import sys, os, io, json, zlib, base64
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner_3_20261203_common import (WATCH, load_prices, load_macro, zscore_series,
                                     cross_sectional_ic, ic_stats, regime_split,
                                     spearman_panel_rho)

ASOF = '2027-02-10'
H = 10
IC_THR, ICIR_THR = 0.0070, 0.0840

px = load_prices(ASOF)
macro = load_macro(ASOF)
fwd = px.shift(-H) / px - 1.0
vix, dxy, usdcny = macro['VIX'], macro['DXY'], macro['USDCNY']
spx, xau, btc, wti = px['SPX'], px['XAU'], px['BTC'], px['WTI']
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


def rskew(s, w, minp=None):
    v = s.dropna()
    if minp is None:
        minp = max(10, int(w * 0.5))
    return v.rolling(w, min_periods=minp).skew().reindex(px.index)


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


spx_r = spx.pct_change()
wti_r = wti.pct_change()
btc_r = btc.pct_change()
xau_r = xau.pct_change()
btc_mom60 = retk(btc, 60)
xau_mom20 = retk(xau, 20)
rv20 = px.apply(lambda s: rstd(s, 20))
rv60 = px.apply(lambda s: rstd(s, 60))


def w1(s, p):
    a = rv20[s].reindex(p.index)
    b = rv60[s].reindex(p.index)
    return ((a - b) / b.replace(0, np.nan)).reindex(p.index)


def w2(s, p):
    c = rcorr(p.pct_change(), spx_r, 60)
    return (c - c.shift(20)).reindex(p.index)


def w3(s, p):
    return xau_mom20.reindex(p.index) - retk(p, 20).reindex(p.index)


def w4(s, p):
    return np.sign(btc_mom60.reindex(p.index).fillna(0)) * rcorr(p.pct_change(), btc_r, 60).reindex(p.index)


def w5(s, p):
    return rskew(p.pct_change(), 60).reindex(p.index)


def w6(s, p):
    # volume z-score: needs per-asset volume; handled outside via vpanel
    return pd.Series(np.nan, index=p.index)


def w7(s, p):
    r = p.pct_change()
    m = (wti_r < 0).reindex(p.index).fillna(False)
    x_r = wti_r.where(m)
    df = pd.concat([r.rename('y'), x_r.rename('x')], axis=1, sort=True).dropna()
    if len(df) < 20:
        return pd.Series(np.nan, index=p.index)
    cov = df['y'].rolling(60, min_periods=15).cov(df['x'])
    var = df['x'].rolling(60, min_periods=15).var()
    return (cov / var).replace([np.inf, -np.inf], np.nan).reindex(p.index)


def w8(s, p):
    lo = p.rolling(60, min_periods=20).min()
    hi = p.rolling(60, min_periods=20).max()
    rng = (hi - lo).replace(0, np.nan)
    return ((p - lo) / rng).reindex(p.index)


factors = {
    'vol_convexity_20x60': build_factor('vol_convexity_20x60', w1),
    'mom_corr_spx_60': build_factor('mom_corr_spx_60', w2),
    'xau_diverg_20': build_factor('xau_diverg_20', w3),
    'btc_trend_corr_60': build_factor('btc_trend_corr_60', w4),
    'ret_skew_60': build_factor('ret_skew_60', w5),
    'wti_down_beta_60': build_factor('wti_down_beta_60', w7),
    'drawup_60': build_factor('drawup_60', w8),
}

# volume_z_20 handled separately (9 assets with volume)
vols = {}
import glob
for s in WATCH:
    f = f'../persistent/stock_data/{s}.csv'
    if not os.path.exists(f):
        continue
    df = pd.read_csv(f, parse_dates=['date'])
    df = df[df['date'] <= pd.Timestamp(ASOF)].sort_values('date')
    if 'volume' in df.columns and df['volume'].astype(float).abs().sum() > 0:
        vols[s] = df.set_index('date')['volume'].astype(float)
vpanel = pd.DataFrame(vols).sort_index()
volz = vpanel / vpanel.rolling(20, min_periods=10).mean() - 1.0
volz = volz.reindex(px.index)
factors['volume_z_20'] = volz.reindex(columns=WATCH)

# ---------------- library panels (fresh recompute) ----------------
def decode_artifact(fn):
    d = json.load(open(fn))
    art = d['validation']['signal_artifact']['data']
    df = pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(art)).decode()))
    df.index = pd.to_datetime(df[df.columns[0]])
    return df.drop(columns=df.columns[0])


lib_panels = {'usdcny_beta_60': decode_artifact('factors/usdcny_beta_60.json')}
lib_panels['mom_10d_skip5'] = px / px.shift(15) - 1.0
vb = {s: rolling_beta(px[s].pct_change(), vix.pct_change(), 60) for s in WATCH}
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame(vb).sort_index()
yb = {s: rolling_beta(px[s].pct_change(), us10.pct_change(), 60) for s in WATCH}
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

json.dump(res, open('scripts/_miner1_20270211_screen_results.json', 'w'), indent=1, default=str, allow_nan=True)

# ---------------- drift re-validation of effective library ----------------
print("\n=== Drift re-validation of active library + fallback ensemble through 2027-02-10 ===")
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
print("\n=== Live snapshot of strongest candidates on 2027-02-10 (z-scored) ===")
order = sorted(res, key=lambda k: -(abs(res[k].get('ic', 0)) if isinstance(res[k].get('ic'), float) else 0))
for name in order[:4]:
    try:
        fz = factors[name].apply(zscore_series, axis=0)
        last = fz.iloc[-1].dropna().sort_values(ascending=False)
        print(f"{name}:")
        print(last.round(2).to_string())
    except Exception as e:
        print(name, 'ERR', e)
print("\nSaved -> scripts/_miner1_20270211_screen_results.json")