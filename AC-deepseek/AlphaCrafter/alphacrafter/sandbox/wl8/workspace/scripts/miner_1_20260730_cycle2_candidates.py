"""miner_1 factor mining cycle 2026-07-30 (batch 2).
Explore NEW factor families with low expected correlation to the existing library
(mom_10d_skip5, mom_120d_skip5, vix_beta_cond_60x20, vol_of_vol20x60).
Uses the shared per-asset dense-calendar validation framework (factor_validation_lib).
Admission gates: |IC|>=0.0070, |ICIR|>=0.0840 at h=10, min 8 assets/date.
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   load_library_panels, max_library_corr,
                                   print_result, IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
macro = {
    "VIX": load_index("VIX"),
    "DXY": load_index("DXY"),
    "USDCNY": load_index("USDCNY"),
    "USDJPY": load_index("USDJPY"),
    "EURUSD": load_index("EURUSD"),
}
print(f"Panel dates {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows, {close.shape[1]} assets")
lib = load_library_panels()
print(f"Library panels loaded: {list(lib.keys())}")

CS_MEAN_20 = close.pct_change(20).mean(axis=1)
CS_MEAN_60 = close.pct_change(60).mean(axis=1)
EW_RET = close.pct_change().mean(axis=1)          # equal-weight portfolio daily return
BTC_RET = close["BTC"].pct_change()

# ---------------- candidate definitions (per-asset dense series) ----------------
def f_rel_mom_20(c, v, o, h, l, m):
    r = c / c.shift(20) - 1.0
    return r - CS_MEAN_20.reindex(c.index)

def f_rel_mom_60(c, v, o, h, l, m):
    r = c / c.shift(60) - 1.0
    return r - CS_MEAN_60.reindex(c.index)

def f_mom_rv_60x20(c, v, o, h, l, m):
    r = c.pct_change()
    mom = c / c.shift(60) - 1.0
    return mom / r.rolling(20).std().clip(lower=1e-9)

def f_sharpe_60(c, v, o, h, l, m):
    r = c.pct_change()
    return r.rolling(60).mean() / r.rolling(60).std().clip(lower=1e-12)

def f_streak_20(c, v, o, h, l, m):
    r = c.pct_change()
    return (r > 0).rolling(20).mean()

def f_trend_slope_60(c, v, o, h, l, m):
    y = np.log(c)
    n = 60
    k = np.arange(len(y), dtype=float)
    sy = y.rolling(n).sum()
    sk = pd.Series(k, index=y.index).rolling(n).sum()
    sk2 = pd.Series(k * k, index=y.index).rolling(n).sum()
    sxy = (y * pd.Series(k, index=y.index)).rolling(n).sum()
    denom = (n * sk2 - sk * sk).clip(lower=1e-12)
    slope = (n * sxy - sk * sy) / denom
    rv = c.pct_change().rolling(n).std().clip(lower=1e-12)
    return slope * n / rv   # trend move over window in vol units

def f_vol_ratio_10x60(c, v, o, h, l, m):
    r = c.pct_change()
    return r.rolling(10).std() / r.rolling(60).std().clip(lower=1e-12)

def f_downside_vol_ratio_60(c, v, o, h, l, m):
    r = c.pct_change()
    neg = r.clip(upper=0)
    ds = (neg ** 2).rolling(60).mean() ** 0.5
    return ds / r.rolling(60).std().clip(lower=1e-12)

def f_semivol_ratio_60(c, v, o, h, l, m):
    r = c.pct_change()
    neg = r.clip(upper=0)
    pos = r.clip(lower=0)
    ds = (neg ** 2).rolling(60).mean() ** 0.5
    us = (pos ** 2).rolling(60).mean() ** 0.5
    return ds / us.clip(lower=1e-12)

def f_skew_60(c, v, o, h, l, m):
    return c.pct_change().rolling(60).skew()

def f_kurt_60(c, v, o, h, l, m):
    return c.pct_change().rolling(60).kurt()

def f_eff_ratio_20(c, v, o, h, l, m):
    r = c.pct_change().abs()
    return (c - c.shift(20)).abs() / r.rolling(20).sum().clip(lower=1e-12)

def f_gap_ratio_20(c, v, o, h, l, m):
    prev_close = c.shift(1)
    gap = (o - prev_close) / prev_close
    return gap.rolling(20).mean()

def _cond_beta(c, m, name, win=60, move_win=20, sign=-1.0):
    x = m[name]
    xr = x.pct_change().reindex(c.index)
    r = c.pct_change()
    varx = xr.rolling(win).var()
    cov = r.rolling(win).cov(xr)
    beta = cov / varx.replace(0, np.nan)
    move = (x / x.shift(move_win) - 1.0).reindex(c.index)
    return sign * beta * move

def f_dxy_beta_cond_60x20(c, v, o, h, l, m):
    return _cond_beta(c, m, "DXY", 60, 20, -1.0)

def f_jpy_beta_cond_60x20(c, v, o, h, l, m):
    return _cond_beta(c, m, "USDJPY", 60, 20, -1.0)

def f_yield_beta_cond_60x20(c, v, o, h, l, m):
    x = m["US10Y"] if "US10Y" in m else None
    # US10Y is a tradable asset, use its own close series for the move
    us10 = close["US10Y"].reindex(c.index)
    r = c.pct_change()
    dx = us10.diff()
    varx = dx.rolling(60).var()
    cov = r.rolling(60).cov(dx)
    beta = cov / varx.replace(0, np.nan)
    move = (us10 / us10.shift(20) - 1.0)
    return beta * move

def f_market_beta_60(c, v, o, h, l, m):
    er = EW_RET.reindex(c.index)
    r = c.pct_change()
    varx = er.rolling(60).var()
    cov = r.rolling(60).cov(er)
    return cov / varx.replace(0, np.nan)

def f_crypto_beta_60(c, v, o, h, l, m):
    br = BTC_RET.reindex(c.index)
    r = c.pct_change()
    varx = br.rolling(60).var()
    cov = r.rolling(60).cov(br)
    return cov / varx.replace(0, np.nan)

CANDIDATES = [
    ("rel_mom_20", f_rel_mom_20, "20d relative momentum (asset minus cross-sectional mean)"),
    ("rel_mom_60", f_rel_mom_60, "60d relative momentum"),
    ("mom_rv_60x20", f_mom_rv_60x20, "60d momentum scaled by 20d vol (risk-adjusted mom)"),
    ("sharpe_60", f_sharpe_60, "60d Sharpe proxy (mean/std of daily ret)"),
    ("streak_20", f_streak_20, "20d positive-day fraction (trend consistency)"),
    ("trend_slope_60", f_trend_slope_60, "60d log-price OLS slope normalized by vol"),
    ("vol_ratio_10x60", f_vol_ratio_10x60, "10d/60d realized vol regime ratio"),
    ("downside_vol_ratio_60", f_downside_vol_ratio_60, "60d downside semideviation / total vol"),
    ("semivol_ratio_60", f_semivol_ratio_60, "60d downside/upside semideviation ratio"),
    ("skew_60", f_skew_60, "60d return skewness"),
    ("kurt_60", f_kurt_60, "60d return kurtosis"),
    ("eff_ratio_20", f_eff_ratio_20, "20d Kaufman efficiency ratio"),
    ("gap_ratio_20", f_gap_ratio_20, "20d mean overnight gap (open/prev close)"),
    ("dxy_beta_cond_60x20", f_dxy_beta_cond_60x20, "-beta(DXY,60) x DXY 20d move"),
    ("jpy_beta_cond_60x20", f_jpy_beta_cond_60x20, "-beta(USDJPY,60) x USDJPY 20d move"),
    ("yield_beta_cond_60x20", f_yield_beta_cond_60x20, "beta(US10Y diff,60) x US10Y 20d move"),
    ("market_beta_60", f_market_beta_60, "60d beta to equal-weight portfolio"),
    ("crypto_beta_60", f_crypto_beta_60, "60d beta to BTC returns"),
]

results = {}
for name, fn, desc in CANDIDATES:
    try:
        res = validate_factor(fn, close, vol, open_, high, low, macro)
        res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
        results[name] = res
        print_result(f"{name} [{desc}]", res)
        print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")
    except Exception as e:
        print(f"[ERROR] {name}: {type(e).__name__}: {e}")

print("\n\n===== SUMMARY =====")
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    print(f"{name:26s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"n={res['n_ic_dates']:4d} cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} libcorr={res['max_abs_library_correlation']:.3f} -> {'PASS' if ok else 'fail'}")

with open("scripts/_miner1_cycle2_results.json", "w") as f:
    json.dump({k: {kk: vv for kk, vv in v.items() if kk != "panel"} for k, v in results.items()},
              f, indent=1, default=str)
print("\nSaved scripts/_miner1_cycle2_results.json")
