# -*- coding: utf-8 -*-
"""miner_2 2027-01-28 cycle: screen novel cross-asset factor candidates (v3).
NaN-aware rolling windows (min_periods ~ w*0.5, NaN dropped inside lambdas).
Visible data through 2027-01-27. Gates (h=10): |IC| >= 0.0070, |ICIR| >= 0.0840.
"""
import sys, json, math
import numpy as np
import pandas as pd

sys.path.insert(0, 'scripts')
import miner3_lib as L

L.LIB_FACTORS = [f for f in L.LIB_FACTORS if f != 'kurt_20'] + ['kurt_20']

VIS = '2027-01-28'
C, V, H, Lo, O = L.load_close_panel(4000)
mask = C.index < VIS
C, V, H, Lo, O = C[mask], V[mask], H[mask], Lo[mask], O[mask]
R = C.pct_change()
LR = np.log(C).diff()
EW = LR.mean(axis=1)

print(f"Panel dates: {C.index.min().date()} -> {C.index.max().date()}, {len(C)} rows")

MP = lambda w: int(w * 0.5)

def _clean(a):
    a = np.asarray(a, dtype=float)
    return a[~np.isnan(a)]

def _apply_win(panel, w, fn):
    return panel.rolling(w, min_periods=MP(w)).apply(
        lambda a: fn(_clean(a)), raw=True)

# ---------- A) trend consistency ----------
def kaufman_er(w=60):
    num = (C - C.shift(w)).abs()
    den = R.abs().rolling(w, min_periods=MP(w)).sum()
    return num / den.replace(0, np.nan)

def macd_slope(w1=12, w2=26, look=10):
    macd = C.ewm(span=w1, adjust=False).mean() - C.ewm(span=w2, adjust=False).mean()
    return (macd - macd.shift(look)) / C.replace(0, np.nan)

def trend_accel_30(w=30):
    lr = np.log(C)
    slope = (lr - lr.shift(w)) / w
    return slope - slope.shift(w)

def range_pos_60(w=60):
    hi = C.rolling(w, min_periods=MP(w)).max()
    lo = C.rolling(w, min_periods=MP(w)).min()
    return (C - lo) / (hi - lo).replace(0, np.nan)

def mom_250d_skip20(w=250, skip=20):
    return C / C.shift(w + skip) - 1.0

# ---------- B) defensive asymmetry ----------
def rolling_cond_beta(asset_r, ref_r, w, cond_series):
    ref_df = pd.DataFrame({c: ref_r for c in asset_r.columns}, index=asset_r.index)
    m = pd.DataFrame({c: cond_series for c in asset_r.columns}, index=asset_r.index)
    a = asset_r.where(m)
    r = ref_df.where(m)
    cov = pd.DataFrame(index=asset_r.index, columns=asset_r.columns, dtype=float)
    for c in asset_r.columns:
        cov[c] = a[c].rolling(w, min_periods=MP(w)).cov(r[c])
    var = r.rolling(w, min_periods=MP(w)).var()
    return cov / var.replace(0, np.nan)

def down_upside_beta_60(w=60):
    down_beta = rolling_cond_beta(R, EW, w, EW < 0)
    up_beta = rolling_cond_beta(R, EW, w, EW > 0)
    return down_beta - up_beta

def down_capture_60(w=60):
    return _apply_win(R, 60, lambda a: a[a < 0].mean() if (a < 0).sum() >= 5 else np.nan)

def profit_factor_60(w=60):
    pos = _apply_win(R, w, lambda a: a[a > 0].mean() if (a > 0).sum() >= 5 else np.nan)
    neg = _apply_win(R, w, lambda a: a[a < 0].mean() if (a < 0).sum() >= 5 else np.nan)
    return pos / neg.abs().replace(0, np.nan)

def win_rate_60(w=60):
    return (R > 0).rolling(w, min_periods=MP(w)).mean()

# ---------- C) vol structure ----------
def rv_ratio_20x60(w1=20, w2=60):
    s1 = R.rolling(w1, min_periods=int(w1 * 0.5)).std(ddof=0)
    s2 = R.rolling(w2, min_periods=int(w2 * 0.5)).std(ddof=0)
    return s1 / s2.replace(0, np.nan)

def vol_asym_60(w=60):
    ds = _apply_win(R, w, lambda a: a[a < 0].std(ddof=0) if (a < 0).sum() >= 5 else np.nan)
    us = _apply_win(R, w, lambda a: a[a > 0].std(ddof=0) if (a > 0).sum() >= 5 else np.nan)
    return ds / us.replace(0, np.nan)

def hurst_vr_60(w=60, k=5):
    v1 = R.rolling(w, min_periods=MP(w)).var(ddof=0)
    vk = R.rolling(w // k, min_periods=int((w // k) * 0.5)).var(ddof=0).rolling(k, min_periods=k).mean()
    vr = v1 / (vk * k).replace(0, np.nan)
    return np.log(vr)

# ---------- D) cross-asset ----------
def def_rel_mom_60(w=60):
    def_basket = LR[['XAU', 'US10Y', 'CN10Y']].mean(axis=1)
    def_cum = def_basket.rolling(w, min_periods=MP(w)).sum()
    asset_cum = LR.rolling(w, min_periods=MP(w)).sum()
    return asset_cum.sub(def_cum, axis=0)

CANDIDATES = {
    'kaufman_er_60': kaufman_er,
    'macd_slope_20x10': macd_slope,
    'trend_accel_30': trend_accel_30,
    'range_pos_60': range_pos_60,
    'mom_250d_skip20': mom_250d_skip20,
    'down_upside_beta_60': down_upside_beta_60,
    'down_capture_60': down_capture_60,
    'profit_factor_60': profit_factor_60,
    'win_rate_60': win_rate_60,
    'rv_ratio_20x60': rv_ratio_20x60,
    'vol_asym_60': vol_asym_60,
    'hurst_vr_60': hurst_vr_60,
    'def_rel_mom_60': def_rel_mom_60,
}

results = {}
for name, fn in CANDIDATES.items():
    try:
        fp = fn()
        fp = fp.replace([np.inf, -np.inf], np.nan)
        s = L.rank_ic(fp, R.shift(-10))
        if s is None or len(s) < 30:
            results[name] = {"error": f"insufficient IC dates ({0 if s is None else len(s)})"}
            print(f"{name}: INSUFFICIENT ({0 if s is None else len(s)} dates)")
            continue
        summ = L.summarize(s, 10, name)
        summ['decay_ic_by_horizon'] = L.decay_analysis(fp, R)
        cov = L.coverage_turnover(fp, R, 10)
        summ.update(cov)
        rhos, maxrho = L.library_max_rho(fp)
        summ['library_rho_by_factor'] = rhos
        summ['max_abs_library_correlation'] = round(maxrho, 3)
        results[name] = summ
        gate = abs(summ['ic']) >= 0.0070 and abs(summ['icir']) >= 0.0840
        print(f"{name}: ic={summ['ic']:.4f} icir={summ['icir']:.4f} n={summ['n_ic_dates']} "
              f"hit={summ['ic_hit_ratio']:.2f} cov_asset={summ['coverage_asset_days']:.3f} "
              f"cov_dates={summ['coverage_dates_ge8']:.3f} rho={summ['max_abs_library_correlation']:.3f} "
              f"decay10={summ['decay_ic_by_horizon'].get('10')} {'*** PASS ***' if gate else 'fail'}")
    except Exception as e:
        results[name] = {"error": str(e)}
        print(f"{name}: ERROR {e}")

with open('scripts/miner_2_20270128_screen_results.json', 'w') as f:
    json.dump(results, f, indent=1, default=str)
print("\nSaved scripts/miner_2_20270128_screen_results.json")
