"""miner_1 cycle 2027-04-08: explore fresh cross-asset candidates on the 15-asset universe.
Data visible through 2027-04-07 (previous completed trading day). No future leakage.

Admission gates (shared, 15-asset universe): |IC_10d| >= 0.0070 and |ICIR_10d| >= 0.0840.
Audit: max_abs_library_correlation vs usdcny_beta_60 (decoded artifact) + recomputed fallback
ensemble panels (mom_10d_skip5, vix_beta_cond_60x20, yield_beta_cond_60x20).

Context: 3 consecutive negative online blocks; momentum whipsaw; VIX re-expansion (feed now
live after 17 sessions at 9.0); rates-pullback tape where defensive XAU/US10Y fell with risk;
yield dir+1 leg burdened by US10Y -6.4% 10d reversal. Candidates target the live-VIX / rate
regime dimensions with faster windows, plus China-complex stabilization plays.

Candidates:
  B1 rate_reversal_gate_20  : 20d momentum x (-1) when US10Y 20d move < 0 (defensive rotation
                              in falling-yield/risk-off tape; disables own momentum)
  B2 us10y_duration_20      : -(20d corr(asset ret, US10Y ret)) (fast duration proxy, live tape)
  B3 vix_beta_20            : 20d unconditional beta to VIX daily returns (direction -1)
  B4 vix_beta_cond_20x5     : 20d beta to VIX on VIX-up days (fast variant of library 60x20)
  B5 hsi_turn_21x5          : sign(HSI 5d reversal) * asset 5d ret (China stabilization follow)
  B6 xau_cop_regime_60      : sign(XAU/COPPER 60d trend) * asset 60d mom (risk-on/off gate)
  B7 jump_reversal_3x20     : -(3d ret)/RV20 (fast vol-scaled reversal after whipsaw)
  B8 downside_ratio_20      : 20d downside capture / RV20 (asymmetry, defensive quality)
  B9 mom_skew_20x60         : (20d mom) * (skew of 20d daily rets) (momentum x lottery-tilt)
  B10 rate_hi_def_20        : -asset 20d ret when US10Y level above 250d median (high-rate
                              regime: fade everything, pure regime defensive)
Also drift re-validation of active library (usdcny_beta_60 artifact + fallback trio).
"""
import sys, os, io, json, zlib, base64
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from miner_1_common import (WATCHLIST, load_panel, load_macro_panel, forward_returns,
                            spearman_ic_series, ic_metrics, regime_slices, zlib_b64_panel)

ASOF = '2027-04-07'
H = 10
IC_THR, ICIR_THR = 0.0070, 0.0840

px, vol = load_panel()
# truncate at ASOF explicitly (load_panel already caps at visible_through; double-check)
px = px[px.index <= pd.Timestamp(ASOF)]
fwd = forward_returns(px, horizon=H)
vix = load_macro_panel('VIX').loc[:ASOF]
us10 = px['US10Y']
print(f"Universe: {len(WATCHLIST)} assets, price dates {px.index[0].date()}..{px.index[-1].date()} ({len(px)} rows)")
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


def build_factor(name, fn):
    cols = {}
    for s in WATCHLIST:
        try:
            cols[s] = fn(s, px[s])
        except Exception:
            cols[s] = np.nan
    return pd.DataFrame(cols).sort_index()


us10_r = us10.pct_change()
us20 = retk(us10, 20)
us_rate_hi = (us10 > us10.rolling(250, min_periods=60).median()).reindex(px.index)
vix_r = vix.pct_change()
vix_up = (vix_r > 0).reindex(px.index)
rv20 = px.apply(lambda s: rstd(s, 20))
rv60 = px.apply(lambda s: rstd(s, 60))
xau_cop = px['XAU'] / px['COPPER']
xau_cop_mom60 = retk(xau_cop, 60)
hsi5 = retk(px['HSI'], 5)
hsi21 = retk(px['HSI'], 21)
hsi_turn = np.sign(-hsi21.reindex(px.index).fillna(0))  # +1 when HSI oversold (neg 21d mom)

# B1 rate_reversal_gate_20: momentum, but flip sign in falling-yield regime
def b1(s, p):
    m = retk(p, 20)
    out = m.where(us20.reindex(p.index) >= 0, -m)
    return out.reindex(p.index)

# B2 us10y_duration_20: -20d corr of asset return with US10Y return
def b2(s, p):
    return (-1.0 * rcorr(p.pct_change(), us10_r, 20)).reindex(p.index)

# B3 vix_beta_20: 20d beta to VIX returns (unconditional, live-VIX window)
def b3(s, p):
    return rolling_beta(p.pct_change(), vix_r, 20).reindex(p.index)

# B4 vix_beta_cond_20x5: 20d beta to VIX on VIX-up days
def b4(s, p):
    return rolling_beta(p.pct_change(), vix_r, 20, cond=vix_up).reindex(p.index)

# B5 hsi_turn_21x5: China stabilization: sign(-HSI 21d) * asset 5d ret
def b5(s, p):
    return (hsi_turn * retk(p, 5)).reindex(p.index)

# B6 xau_cop_regime_60: sign(XAU/COPPER 60d trend) * asset 60d mom
def b6(s, p):
    return (np.sign(xau_cop_mom60.reindex(p.index).fillna(0)) * retk(p, 60)).reindex(p.index)

# B7 jump_reversal_3x20: -(3d ret)/RV20
def b7(s, p):
    return (-retk(p, 3) / rv20[s].reindex(p.index)).reindex(p.index)

# B8 downside_ratio_20: mean of negative daily returns (20d) / RV20
def b8(s, p):
    r = p.pct_change()
    dneg = r.where(r < 0).rolling(20, min_periods=8).mean()
    return (dneg / rv20[s].reindex(p.index)).reindex(p.index)

# B9 mom_skew_20x60: 20d momentum x rolling skew of 20d daily rets
def b9(s, p):
    r = p.pct_change()
    sk = r.rolling(20, min_periods=10).skew()
    return (retk(p, 20) * sk).reindex(p.index)

# B10 rate_hi_def_20: pure regime defensive - fade all assets in high-rate regime
def b10(s, p):
    return (-retk(p, 20) * rate_hi).reindex(p.index)


rate_hi = us_rate_hi.astype(float).reindex(px.index)

factors = {
    'rate_reversal_gate_20': build_factor('rate_reversal_gate_20', b1),
    'us10y_duration_20': build_factor('us10y_duration_20', b2),
    'vix_beta_20': build_factor('vix_beta_20', b3),
    'vix_beta_cond_20x5': build_factor('vix_beta_cond_20x5', b4),
    'hsi_turn_21x5': build_factor('hsi_turn_21x5', b5),
    'xau_cop_regime_60': build_factor('xau_cop_regime_60', b6),
    'jump_reversal_3x20': build_factor('jump_reversal_3x20', b7),
    'downside_ratio_20': build_factor('downside_ratio_20', b8),
    'mom_skew_20x60': build_factor('mom_skew_20x60', b9),
    'rate_hi_def_20': build_factor('rate_hi_def_20', b10),
}

# ---------------- library panels (fresh recompute) ----------------
def decode_artifact(fn):
    d = json.load(open(fn))
    art = d['validation']['signal_artifact']['data']
    df = pd.read_csv(io.StringIO(zlib.decompress(base64.b64decode(art)).decode()))
    df.index = pd.to_datetime(df[df.columns[0]])
    return df.drop(columns=df.columns[0])

lib_panels = {}
try:
    lib_panels['usdcny_beta_60'] = decode_artifact('factors/usdcny_beta_60.json')
except Exception as e:
    print('artifact decode err', e)
lib_panels['mom_10d_skip5'] = px / px.shift(15) - 1.0
vb = {s: rolling_beta(px[s].pct_change(), vix_r, 60, cond=(vix_r > 0)) for s in WATCHLIST}
lib_panels['vix_beta_cond_60x20'] = pd.DataFrame(vb).sort_index()
yb = {s: rolling_beta(px[s].pct_change(), us10_r, 60) * (us20.reindex(px.index)) for s in WATCHLIST}
lib_panels['yield_beta_cond_60x20'] = pd.DataFrame(yb).sort_index()

# ---------------- evaluate candidates ----------------
res = {}
for name, f in factors.items():
    try:
        fz = f.apply(lambda x: (x - x.median()) / (1.4826 * (x - x.median()).abs().median() + 1e-12)).clip(-5, 5)
        icd = spearman_ic_series(fz, fwd, min_assets=8)
        st = ic_metrics(icd)
        ranks = fz.rank(axis=1)
        to = ranks.diff(10).abs().mean().mean() / (len(WATCHLIST) - 1)
        n_valid = int(fz.notna().sum().sum())
        cov = n_valid / (len(fz) * len(WATCHLIST))
        rhos = {}
        for lname, lp in lib_panels.items():
            lpc = lp.reindex(fz.index)
            common = fz.index.intersection(lpc.index)
            rr = []
            for dt in common:
                x = fz.loc[dt]; y = lpc.loc[dt].reindex(x.index)
                m = x.notna() & y.notna()
                if m.sum() < 8:
                    continue
                xx, yy = x[m], y[m]
                if xx.nunique() < 3 or yy.nunique() < 3:
                    continue
                r = xx.rank().corr(yy.rank())
                if not np.isnan(r):
                    rr.append(r)
            rhos[lname] = float(np.mean(rr)) if rr else float('nan')
        maxrho = max([abs(v) for v in rhos.values() if v == v], default=0.0)
        reg = regime_slices(icd)
        decay = {}
        for hh in [1, 2, 3, 5, 10, 20]:
            fh = forward_returns(px, horizon=hh)
            icd_h = spearman_ic_series(fz, fh, min_assets=8)
            decay[hh] = float(ic_metrics(icd_h)['ic']) if len(icd_h) else np.nan
        gate = (abs(st['ic']) >= IC_THR) and (abs(st['icir']) >= ICIR_THR) and st['n_ic_dates'] >= 60
        res[name] = {'ic': st['ic'], 'icir': st['icir'], 'hit': st['hit'], 'n_dates': st['n_ic_dates'],
                     'turnover_10d': to, 'coverage': cov,
                     'rho_lib': {k: (round(v, 4) if v == v else None) for k, v in rhos.items()},
                     'max_lib_rho': maxrho, 'regime': reg, 'decay': decay,
                     'flag': 'PASS' if gate else 'fail'}
        print(f"== {name} == {'PASS' if gate else 'fail'}")
        print(f"   ic={st['ic']:.4f} icir={st['icir']:.4f} hit={st['hit']:.3f} n_dates={st['n_ic_dates']}")
        print(f"   turnover10d={to:.3f} coverage={cov:.3f} max_lib_rho={maxrho:.3f} "
              f"rhos={ {k: v for k, v in rhos.items()} }")
        print(f"   regime={ {k: [round(x, 4) for x in v] if isinstance(v, list) else v for k, v in reg.items()} }")
        print(f"   decay={ {k: (round(v, 4) if v == v else None) for k, v in decay.items()} }")
    except Exception as e:
        res[name] = {'error': str(e), 'flag': 'err'}
        print(f"== {name} == ERROR: {e}")

json.dump(res, open('scripts/_miner1_20270408_screen_results.json', 'w'), indent=1, default=str, allow_nan=True)

# ---------------- drift re-validation of active library ----------------
print("\n=== Drift re-validation of active library + fallback ensemble through 2027-04-07 ===")
combo = {'mom_10d_skip5': lib_panels['mom_10d_skip5'],
         'vix_beta_cond_60x20': lib_panels['vix_beta_cond_60x20'],
         'yield_beta_cond_60x20': lib_panels['yield_beta_cond_60x20']}
for name, f in combo.items():
    fz = f.apply(lambda x: (x - x.median()) / (1.4826 * (x - x.median()).abs().median() + 1e-12)).clip(-5, 5)
    full = ic_metrics(spearman_ic_series(fz, fwd, min_assets=8))
    sl = fz.iloc[-90:]; fl = fwd.reindex(sl.index)
    last_90 = ic_metrics(spearman_ic_series(sl, fl, min_assets=8))
    s2 = fz.iloc[-180:]; f2 = fwd.reindex(s2.index)
    last_180 = ic_metrics(spearman_ic_series(s2, f2, min_assets=8))
    print(f"{name}: full ic={full['ic']:.4f} icir={full['icir']:.4f} n={full['n_ic_dates']} | "
          f"90d ic={last_90['ic']:.4f} icir={last_90['icir']:.4f} n={last_90['n_ic_dates']} | "
          f"180d ic={last_180['ic']:.4f} icir={last_180['icir']:.4f} n={last_180['n_ic_dates']}")

try:
    uz = decode_artifact('factors/usdcny_beta_60.json')
    uz = uz.apply(lambda x: (x - x.median()) / (1.4826 * (x - x.median()).abs().median() + 1e-12)).clip(-5, 5)
    full = ic_metrics(spearman_ic_series(uz, fwd, min_assets=8))
    sl = uz.iloc[-90:]; fl = fwd.reindex(sl.index)
    last_90 = ic_metrics(spearman_ic_series(sl, fl, min_assets=8))
    s2 = uz.iloc[-180:]; f2 = fwd.reindex(s2.index)
    last_180 = ic_metrics(spearman_ic_series(s2, f2, min_assets=8))
    print(f"usdcny_beta_60 (artifact): full ic={full['ic']:.4f} icir={full['icir']:.4f} n={full['n_ic_dates']} | "
          f"90d ic={last_90['ic']:.4f} icir={last_90['icir']:.4f} n={last_90['n_ic_dates']} | "
          f"180d ic={last_180['ic']:.4f} icir={last_180['icir']:.4f} n={last_180['n_ic_dates']}")
except Exception as e:
    print('usdcny artifact revalidation err', e)