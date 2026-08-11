# -*- coding: utf-8 -*-
"""miner_2 2026-12-31 cycle: screen novel cross-asset factor candidates.
Visible data through 2026-12-30 (current date 2026-12-31).

Admission gates on the 15-instrument cross-asset universe (h=10):
|IC| >= 0.0070 and |ICIR| >= 0.0840.

Families (all chosen to be novel vs library: trend_r2_30_signed,
semi_down_ratio_20, mom_120d_skip5, mom_10d_skip5, vol_of_vol20x60,
dxy_beta_60, WTI_BETA_60, vix_beta_cond_60x20, time_under_water_120, kurt_20):
  A) volume/liquidity: amihud_illiq_20, vol_price_corr_20, vol_zscore_20
  B) vol structure:    atr_ratio_20x60, parkinson_eff_60, downside_vol_ratio_20,
                       range_squeeze_20x60
  C) return decompos:  overnight_share_20, gap_freq_20, intraday_mom_20
  D) momentum quality: resid_mom_20_spx, rel_strength_60_ew, sortino_60,
                       vol_conf_mom_20x60
  E) macro-conditional: dxy_mom_beta_20, vix_up_beta_20, yield_mom_beta_60,
                       spx_beta_regime_60
"""
import sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import miner3_lib as L

# library correlation set: include kurt_20 (persisted library factor)
L.LIB_FACTORS = [f for f in L.LIB_FACTORS if f != 'kurt_20'] + ['kurt_20']

VIS = '2026-12-31'
C, V, H, Lo, O = L.load_close_panel(4000)
mask = C.index < VIS
C, V, H, Lo, O = C[mask], V[mask], H[mask], Lo[mask], O[mask]
R = C.pct_change()
LR = np.log(C).diff()
EW = LR.mean(axis=1)  # equal-weight cross-asset log-return basket

def load_macro(name):
    df = pd.read_csv(f'../persistent/index_data/{name}.csv', parse_dates=['date'])
    df['date'] = pd.to_datetime(df['date']).dt.normalize()
    df = df.set_index('date').sort_index()
    df = df[df.index < VIS]
    return df

DXY = load_macro('DXY'); VIX = load_macro('VIX'); US10Y_M = load_macro('USDJPY')  # placeholder

def macro_ret(name):
    df = load_macro(name)
    s = df['close'] if 'close' in df.columns else df.iloc[:, 1]
    s = s.astype(float)
    return np.log(s).diff()

DXY_R = macro_ret('DXY')
VIX_R = macro_ret('VIX')
US10Y_R = macro_ret('US10Y') if False else None
# US10Y tradable close panel is the 10y yield proxy
US10Y_R = LR['US10Y']

def rolling_beta(asset_ret, ref_ret, win):
    cov = asset_ret.rolling(win).cov(ref_ret)
    var = ref_ret.rolling(win).var()
    return cov / var

# ---------- A) volume / liquidity ----------
def amihud_illiq(w=20):
    ill = (R.abs() / V.replace(0, np.nan))
    return ill.rolling(w).mean()

def vol_price_corr(w=20):
    return R.rolling(w).corr(V)

def vol_zscore(w=20, base=60):
    vm = V.rolling(w).mean()
    vb = V.rolling(base).mean()
    vs = V.rolling(base).std()
    return (vm - vb) / vs.replace(0, np.nan)

# ---------- B) volatility structure ----------
def atr_ratio(w1=20, w2=60):
    pc = C.shift(1)
    tr = pd.concat([(H - Lo), (H - pc).abs(), (Lo - pc).abs()], axis=1)
    tr = tr.groupby(tr.index).max()  # elementwise max across the 3 cols per date
    a1 = tr.rolling(w1).mean()
    a2 = tr.rolling(w2).mean()
    return a1 / a2.replace(0, np.nan)

def parkinson_eff(w=60):
    rv = R.rolling(w).std(ddof=0) * np.sqrt(252)
    hl = np.log(H / Lo)
    pv = np.sqrt((hl ** 2).rolling(w).mean() / (4 * np.log(2))) * np.sqrt(252)
    return rv / pv.replace(0, np.nan)

def downside_vol_ratio(w=20):
    ds = (np.minimum(R, 0) ** 2).rolling(w).mean() ** 0.5
    ts = (R ** 2).rolling(w).mean() ** 0.5
    return ds / ts.replace(0, np.nan)

def range_squeeze(w1=20, w2=60):
    r1 = (C.rolling(w1).max() - C.rolling(w1).min())
    r2 = (C.rolling(w2).max() - C.rolling(w2).min())
    return r1 / r2.replace(0, np.nan)

# ---------- C) return decomposition ----------
def overnight_share(w=20):
    on = np.log(O / C.shift(1))
    id_ = np.log(C / O)
    cum_on = on.rolling(w).sum()
    cum_id = id_.rolling(w).sum()
    denom = cum_on.abs() + cum_id.abs()
    return cum_on / denom.replace(0, np.nan)

def gap_freq(w=20, thr=0.004):
    g = (O / C.shift(1) - 1.0).abs() > thr
    return g.rolling(w).mean()

def intraday_mom(w=20, skip=5):
    id_ = np.log(C / O)
    return id_.shift(skip).rolling(w).sum()

# ---------- D) momentum quality ----------
def resid_mom(w_reg=60, w_mom=20, skip=5):
    spx = R['SPX']
    beta = rolling_beta(R, spx, w_reg)
    resid = R - beta.mul(spx, axis=0)
    return resid.shift(skip).rolling(w_mom).sum()

def rel_strength(w=60):
    asset_ret = np.log(C / C.shift(w))
    ew_ret = asset_ret.mean(axis=1)
    return asset_ret.sub(ew_ret, axis=0)

def sortino(w=60):
    mu = R.rolling(w).mean()
    ds = (np.minimum(R, 0) ** 2).rolling(w).mean() ** 0.5
    return mu / ds.replace(0, np.nan)

def vol_conf_mom(w=20, skip=5):
    m = np.log(C / C.shift(w))
    v20 = R.rolling(20).std()
    v60 = R.rolling(60).std()
    gate = (v20 > v60).astype(float)
    return m.shift(skip) * gate

# ---------- E) macro-conditional ----------
def regime_beta(ref_ret, w_beta=60, w_mom=20):
    beta = rolling_beta(R, ref_ret, w_beta)
    sign = np.sign(ref_ret.rolling(w_mom).sum()).replace(0, 1.0)
    return beta.mul(sign, axis=0)

cands = [
    ("amihud_illiq_20", amihud_illiq(20), "A"),
    ("vol_price_corr_20", vol_price_corr(20), "A"),
    ("vol_zscore_20", vol_zscore(20, 60), "A"),
    ("atr_ratio_20x60", atr_ratio(20, 60), "B"),
    ("parkinson_eff_60", parkinson_eff(60), "B"),
    ("downside_vol_ratio_20", downside_vol_ratio(20), "B"),
    ("range_squeeze_20x60", range_squeeze(20, 60), "B"),
    ("overnight_share_20", overnight_share(20), "C"),
    ("gap_freq_20", gap_freq(20, 0.004), "C"),
    ("intraday_mom_20", intraday_mom(20, 5), "C"),
    ("resid_mom_20_spx", resid_mom(60, 20, 5), "D"),
    ("rel_strength_60_ew", rel_strength(60), "D"),
    ("sortino_60", sortino(60), "D"),
    ("vol_conf_mom_20x60", vol_conf_mom(20, 5), "D"),
    ("dxy_mom_beta_20", regime_beta(DXY_R, 60, 20), "E"),
    ("vix_up_beta_20", regime_beta(VIX_R, 60, 20), "E"),
    ("yield_mom_beta_60", regime_beta(US10Y_R, 60, 20), "E"),
    ("spx_beta_regime_60", regime_beta(R['SPX'], 60, 20), "E"),
]

print(f"=== SCREEN {VIS} (data through {C.index.max().date()}, n_dates={len(C)}, n_assets={len(C.columns)}) ===")
print(f"gates: |IC|>=0.0070, |ICIR|>=0.0840 @ h=10\n")
hdr = f"{'candidate':22s} {'fam':3s} {'IC':>7s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'covA':>5s} {'cov8':>5s} {'turn':>5s} {'maxrho':>6s} {'pass'}"
print(hdr)
results = {}
for name, fac, fam in cands:
    try:
        summ = L.full_validate(fac, R, horizon=10, label=name)
        m = summ
        ic, icir = m['ic'], m['icir']
        hit, n = m['ic_hit_ratio'], m['n_ic_dates']
        covA, cov8, turn = m['coverage_asset_days'], m['coverage_dates_ge8'], m['turnover_10d_rank']
        mr = m.get('max_abs_library_correlation', 0.0)
        passed = (abs(ic) >= 0.0070 and abs(icir) >= 0.0840 and n >= 100)
        results[name] = {'family': fam, **m}
        print(f"{name:22s} {fam:3s} {ic:+7.4f} {icir:+7.3f} {hit:5.3f} {n:5d} {covA:5.2f} {cov8:5.2f} {turn:5.2f} {mr:+6.3f} {'YES' if passed else 'no'}")
    except Exception as e:
        print(f"{name:22s} ERROR {e}")

with open('scripts/miner_2_20261231_screen_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nsaved -> scripts/miner_2_20261231_screen_results.json")
