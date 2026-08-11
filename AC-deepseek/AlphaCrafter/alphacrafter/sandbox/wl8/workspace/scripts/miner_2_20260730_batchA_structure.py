"""miner_2 2026-07-30 -- batch A: price-structure & novel conditional-beta factors.
Explores factor families NOT yet in the library (library: mom_10d_skip5,
vix_beta_cond_60x20, yield_beta_cond_60x20):
  1. range_pos_10/20  : intraday close location (close-low)/(high-low) - 0.5
  2. gap_10/20        : mean overnight gap open_t/close_{t-1} - 1
  3. accel_10x20      : momentum acceleration log(c/c10) - log(c10/c20)
  4. dd_60/120        : drawdown depth close/rolling_max(close,win) - 1
  5. xau/wti_beta_cond: beta to tradable anchor (XAU/WTI) x 20d anchor move
  6. mom_rv_10x20     : risk-adjusted momentum (mom_10d / vol_20d)
Validation via shared factor_validation_lib (15-asset cross-section, min 8 assets/date,
admission horizon 10d). Prints max_abs_library_correlation for each candidate.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   load_library_panels, max_library_corr,
                                   print_result, IC_GATE, ICIR_GATE)

close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
# tradable anchors for conditional beta (from the tradable universe itself)
macro["XAU"] = close["XAU"].dropna()
macro["WTI"] = close["WTI"].dropna()
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets")
lib = load_library_panels()
print(f"Library: {list(lib.keys())}")


def _beta(asset_ret, macro_ret, win):
    cov = asset_ret.rolling(win).cov(macro_ret)
    var = macro_ret.rolling(win).var()
    return cov / var


def _m(c, m, key):
    return m[key].reindex(c.index).ffill()


# ---- 1. intraday close location ----
def make_range_pos(win):
    def fn(c, v, o, h, l, m, win=win):
        pos = (c - l) / (h - l) - 0.5
        return pos.rolling(win).mean()
    fn.__name__ = f"range_pos_{win}"
    return fn, f"{win}d mean intraday close position in [low,high] (demeaned)"


# ---- 2. overnight gap ----
def make_gap(win):
    def fn(c, v, o, h, l, m, win=win):
        gap = o / c.shift(1) - 1.0
        return gap.rolling(win).mean()
    fn.__name__ = f"gap_{win}"
    return fn, f"{win}d mean overnight gap (open vs prev close)"


# ---- 3. momentum acceleration ----
def f_accel(c, v, o, h, l, m, s=10, lw=20):
    r_s = np.log(c / c.shift(s))
    r_l = np.log(c.shift(s) / c.shift(lw))
    return r_s - r_l


# ---- 4. drawdown depth ----
def make_dd(win):
    def fn(c, v, o, h, l, m, win=win):
        return c / c.rolling(win, min_periods=win // 2).max() - 1.0
    fn.__name__ = f"dd_{win}"
    return fn, f"{win}d drawdown depth close/rolling_max-1 (negative=underwater)"


# ---- 5. tradable-anchor conditional beta ----
def make_anchor_cond(key, win=60, look=20):
    def fn(c, v, o, h, l, m, win=win, look=look):
        ms = _m(c, m, key)
        mr = ms.pct_change()
        b = _beta(c.pct_change(), mr, win)
        move = ms / ms.shift(look) - 1.0
        return b * move
    fn.__name__ = f"{key.lower()}_beta_cond_{win}x{look}"
    return fn, f"{win}d beta to {key} x {look}d {key} move"


# ---- 6. risk-adjusted momentum ----
def f_mom_rv(c, v, o, h, l, m, s=10, w=20):
    r = c.pct_change(s)
    vol = c.pct_change().rolling(w).std()
    return r / vol


cands = [
    ("range_pos_10", *make_range_pos(10)),
    ("range_pos_20", *make_range_pos(20)),
    ("gap_10", *make_gap(10)),
    ("gap_20", *make_gap(20)),
    ("accel_10x20", f_accel, "momentum acceleration: 10d ret - prior 10d ret (log)"),
    ("dd_60", *make_dd(60)),
    ("dd_120", *make_dd(120)),
    ("xau_beta_cond_60x20", *make_anchor_cond("XAU")),
    ("wti_beta_cond_60x20", *make_anchor_cond("WTI")),
    ("mom_rv_10x20", f_mom_rv, "10d momentum / 20d vol (risk-adjusted momentum)"),
]

results = {}
for name, fn, desc in cands:
    res = validate_factor(fn, close, vol, open_, high, low, macro)
    res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
    results[name] = res
    print_result(f"{name} [{desc}]", res)
    print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")

print("\n===== SUMMARY ===== (gate: |IC|>=%.4f and |ICIR|>=%.4f; libcorr<0.5)" % (IC_GATE, ICIR_GATE))
for name, res in results.items():
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    okc = ok and res["max_abs_library_correlation"] < 0.5
    print(f"{name:22s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} n={res['n_ic_dates']} libcorr={res['max_abs_library_correlation']:.3f} "
          f"-> {'PASS' if okc else ('IC-PASS' if ok else 'fail')}")
