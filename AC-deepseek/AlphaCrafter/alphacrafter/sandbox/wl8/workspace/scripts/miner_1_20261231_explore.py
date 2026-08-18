"""miner_1 cycle 2026-12-31: explore fresh cross-asset candidates on the 15-asset universe.
Data visible through 2026-12-30 (previous completed trading day).

Admission gates (shared, 15-asset universe): |IC_10d| >= 0.0070 and |ICIR_10d| >= 0.0840.
Audit: max_abs_library_correlation vs usdcny_beta_60 (decoded artifact) + recomputed ensemble panels.

Candidates:
  F1 fx_cond_mom_20        : 20d momentum * sign(USDCNY 20d move)
  F2 fx_cond_mom_60        : 60d skip5 momentum * sign(USDCNY 60d move)
  F3 yuan_strength_beta_60 : -beta(ret,USDCNY,60)
  F4 yuan_strength_cond_20 : -beta(ret,USDCNY,60) * (USDCNY 20d move)
  F5 dxy_down_corr_20      : corr(ret,DXY,20) on DXY-down days
  F6 tail_hedge_value_20   : 20d ret * z(VIX 20d chg)
  F7 gold_beat_20          : 20d ret - XAU 20d ret
  F8 btc_rv_ratio_20       : BTC 20d RV / asset 20d RV
  F9 spx_stress_beta_60    : corr(ret,SPX,60)|SPX-down - corr|SPX-up
  F10 cn_resid_mom_20      : asset 20d mom - beta*CN 20d mom
Also drift re-validation of effective library factors through 2026-12-30.
"""
import sys, os, io, json, zlib, base64
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner_3_20261203_common import (WATCH, load_prices, load_macro, zscore_series,
                                     cross_sectional_ic, ic_stats, regime_split,
                                     spearman_panel_rho)

ASOF = '2026-12-30'
H = 10
IC_THR, ICIR_THR = 0.0070, 0.0840

px = load_prices(ASOF)
macro = load_macro(ASOF)
fwd = px.shift(-H) / px - 1.0
vix, dxy, usdcny = macro['VIX'], macro['DXY'], macro['USDCNY']
cn = px['000300.SH']
xau, btc, spx = px['XAU'], px['BTC'], px['SPX']

MINP_FRAC = 0.6


def _masked_roll(y, x, w, mask, minp):
    """Rolling beta & corr of y on x over window w using only days where mask is True.
    Returns (beta, corr)."""
    ym, xm = y.where(mask), x.where(mask)
    yxm = (y * x).where(mask)
    y2m = (y * y).where(mask)
    x2m = (x * x).where(mask)
    n = mask.rolling(w).sum()
    sy = ym.rolling(w).sum(); sx = xm.rolling(w).sum()
    syx = yxm.rolling(w).sum(); sy2 = y2m.rolling(w).sum(); sx2 = x2m.rolling(w).sum()
    my = sy / n; mx = sx / n
    cov = syx / n - my * mx
    vy = sy2 / n - my * my
    vx = sx2 / n - mx * mx
    ok = (n >= minp)
    beta = (cov / vx).where(ok & (vx > 0))
    corr = (cov / np.sqrt(vy * vx)).where(ok & (vy > 0) & (vx > 0))
    return beta.replace([np.inf, -np.inf], np.nan), corr.replace([np.inf, -np.inf], np.nan)


def rolling_beta(y, x, w, cond=None, minp=None):
    mask = pd.Series(True, index=y.index) if cond is None else cond.reindex(y.index).fillna(False).astype(bool)
    yv = y.reindex(mask.index).astype(float)
    xv = x.reindex(mask.index).astype(float)
    minp = minp if minp is not None else int(w * MINP_FRAC)
    b, _ = _masked_roll(yv, xv, w, mask, minp)
    return b


def rolling_corr(y, x, w, cond=None, minp=None):
    mask = pd.Series(True, index=y.index) if cond is None else cond.reindex(y.index).fillna(False).astype(bool)
    yv = y.reindex(mask.index).astype(float)
    xv = x.reindex(mask.index).astype(float)
    minp = minp if minp is not None else int(w * MINP_FRAC)
    _, c = _masked_roll(yv, xv, w, mask, minp)
    return c


def build_factor(name, fn):
    cols = {}
    for s in WATCH:
        try:
            cols[s] = fn(s, px[s])
        except Exception:
            cols[s] = np.nan
    return pd.DataFrame(cols).sort_index()


usdcny_chg20 = usdcny / usdcny.shift(20) - 1.0
usdcny_chg60 = usdcny / usdcny.shift(60) - 1.0
dxy_down = dxy.pct_change() < -0.002
vix_z = zscore_series((vix / vix.shift(20) - 1.0))
xau_r20 = xau / xau.shift(20) - 1.0
btc_rv20 = btc.pct_change().rolling(20).std()
spx_r = spx.pct_change()
spx_down = spx_r < 0
spx_up = spx_r > 0
cn_r20 = cn / cn.shift(20) - 1.0


def f1(s, p):
    return (p / p.shift(20) - 1.0) * np.sign(usdcny_chg20.reindex(p.index))


def f2(s, p):
    return (p / p.shift(65) - 1.0) * np.sign(usdcny_chg60.reindex(p.index))


def f3(s, p):
    return -rolling_beta(p.pct_change(), usdcny.pct_change(), 60)


def f4(s, p):
    return -rolling_beta(p.pct_change(), usdcny.pct_change(), 60) * usdcny_chg20.reindex(p.index)


def f5(s, p):
    return rolling_corr(p.pct_change(), dxy.pct_change(), 20, cond=dxy_down)


def f6(s, p):
    return (p / p.shift(20) - 1.0) * vix_z.reindex(p.index)


def f7(s, p):
    return (p / p.shift(20) - 1.0) - xau_r20.reindex(p.index)


def f8(s, p):
    return btc_rv20.reindex(p.index) / p.pct_change().rolling(20).std()


def f9(s, p):
    c_dn = rolling_corr(p.pct_change(), spx_r, 60, cond=spx_down)
    c_up = rolling_corr(p.pct_change(), spx_r, 60, cond=spx_up)
    return c_dn - c_up


def f10(s, p):
    r20 = p / p.shift(20) - 1.0
    b = rolling_beta(p.pct_change(), cn.pct_change(), 60)
    return r20 - b * cn_r20.reindex(p.index)


factors = {
    'fx_cond_mom_20': build_factor('fx_cond_mom_20', f1),
    'fx_cond_mom_60': build_factor('fx_cond_mom_60', f2),
    'yuan_strength_beta_60': build_factor('yuan_strength_beta_60', f3),
    'yuan_strength_cond_20': build_factor('yuan_strength_cond_20', f4),
    'dxy_down_corr_20': build_factor('dxy_down_corr_20', f5),
    'tail_hedge_value_20': build_factor('tail_hedge_value_20', f6),
    'gold_beat_20': build_factor('gold_beat_20', f7),
    'btc_rv_ratio_20': build_factor('btc_rv_ratio_20', f8),
    'spx_stress_beta_60': build_factor('spx_stress_beta_60', f9),
    'cn_resid_mom_20': build_factor('cn_resid_mom_20', f10),
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

ycond20 = (px['US10Y'] / px['US10Y'].shift(20) - 1.0) > 0
vb = {}
for s in WATCH:
    vb[s] = rolling_beta(px[s].pct_change(), vix.pct_change(), 60, cond=ycond20, minp=24)
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame(vb).sort_index()

us10 = px['US10Y']
ycond = (us10 / us10.shift(20) - 1.0) > 0
yb = {}
for s in WATCH:
    yb[s] = rolling_beta(px[s].pct_change(), us10.pct_change(), 60, cond=ycond, minp=24)
lib_panels['yield_beta_cond_60x20'] = pd.DataFrame(yb).sort_index()

# ---------------- evaluate candidates ----------------
print(f"Universe: {len(WATCH)} assets, price dates {px.index[0].date()}..{px.index[-1].date()} ({len(px)} rows)")
print(f"Admission gates: |IC|>={IC_THR}, |ICIR|>={ICIR_THR}, horizon {H}d, min_assets>=8\n")
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
              f"rhos={ {k: (round(v,3) if v==v else None) for k,v in rhos.items()} }")
        print(f"   regime={ {k: [round(x,4) for x in v] for k,v in reg.items()} }")
        print(f"   decay={ {k: round(v,4) if v==v else None for k,v in decay.items()} }")
    except Exception as e:
        res[name] = {'error': str(e), 'flag': 'err'}
        print(f"== {name} == ERROR: {e}")

json.dump(res, open('scripts/_miner1_20261231_screen_results.json', 'w'), indent=1, default=str, allow_nan=True)

# ---------------- drift re-validation of effective library ----------------
print("\n=== Drift re-validation of library factors through 2026-12-30 ===")
combo = {'mom_10d_skip5': lib_panels['mom_10d_skip5'],
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

uz = decode_usdcny().apply(zscore_series, axis=0).reindex(fwd.index)
st_u = ic_stats(cross_sectional_ic(uz, fwd, min_assets=8))
print(f"usdcny_beta_60 (artifact): full ic={st_u['ic']:.4f} icir={st_u['icir']:.4f} n={st_u['n_dates']}")

# ---------------- live snapshot 2026-12-30 ----------------
print("\n=== Live snapshot of strongest candidates on 2026-12-30 (z-scored) ===")
for name in ['gold_beat_20', 'yuan_strength_beta_60', 'fx_cond_mom_60']:
    try:
        fz = factors[name].apply(zscore_series, axis=0)
        last = fz.iloc[-1].dropna().sort_values(ascending=False)
        print(f"{name}:")
        print(last.round(2).to_string())
    except Exception as e:
        print(name, 'ERR', e)
print("\nSaved -> scripts/_miner1_20261231_screen_results.json")