"""miner_2 2026-07-30 -- screen new factor families.
Families: (A) intraday price structure (close location, body, shadows),
(B) overnight gaps, (C) momentum acceleration, (D) drawdown depth,
(E) tradable-asset-anchored conditional betas, (F) vol-adjusted momentum,
(G) range compression / volume trend.
Admission horizon = 10d. Gate: |IC|>=0.007, |ICIR|>=0.084, libcorr<0.5.
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, validate_factor,
                                   load_library_panels, max_library_corr,
                                   print_result, IC_GATE, ICIR_GATE, ASSETS)

close, vol, open_, high, low = load_closes()
macro = {k: load_index(k) for k in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]}
# add tradable anchors into the macro namespace so factors can reference them
for anchor in ["SPX", "XAU", "BTC", "WTI", "NDX", "US10Y"]:
    macro[anchor] = close[anchor].dropna()
print(f"Panel {close.index[0].date()}..{close.index[-1].date()}, {len(close)} rows x {close.shape[1]} assets")
lib = load_library_panels()
print(f"Library: {list(lib.keys())}")


def _safe_div(a, b):
    with np.errstate(divide="ignore", invalid="ignore"):
        return a / b


# ---------- (A) intraday price structure ----------
def f_intraday_pos_20(c, v, o, h, l, m, win=20):
    rng = (h - l).replace(0, np.nan)
    pos = _safe_div(c - l, rng)
    return pos.rolling(win).mean()


def f_body_ratio_20(c, v, o, h, l, m, win=20):
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    return _safe_div(body, rng).rolling(win).mean()


def f_upper_shadow_20(c, v, o, h, l, m, win=20):
    rng = (h - l).replace(0, np.nan)
    us = (h - np.maximum(o, c))
    return _safe_div(us, rng).rolling(win).mean()


def f_lower_shadow_20(c, v, o, h, l, m, win=20):
    rng = (h - l).replace(0, np.nan)
    ls = (np.minimum(o, c) - l)
    return _safe_div(ls, rng).rolling(win).mean()


# ---------- (B) overnight gaps ----------
def f_overnight_gap_20(c, v, o, h, l, m, win=20):
    gap = _safe_div(o, c.shift(1)) - 1.0
    return gap.rolling(win).mean()


def f_overnight_gap_60(c, v, o, h, l, m, win=60):
    gap = _safe_div(o, c.shift(1)) - 1.0
    return gap.rolling(win).mean()


def f_overnight_gap_vol_20(c, v, o, h, l, m, win=20):
    """20d mean abs overnight gap (gap size, not sign)."""
    gap = _safe_div(o, c.shift(1)) - 1.0
    return gap.abs().rolling(win).mean()


# ---------- (C) momentum acceleration ----------
def f_mom_accel_10x20(c, v, o, h, l, m):
    r10 = c / c.shift(10) - 1.0
    r20 = c / c.shift(20) - 1.0
    return r10 - r20


def f_mom_accel_20x60(c, v, o, h, l, m):
    r20 = c / c.shift(20) - 1.0
    r60 = c / c.shift(60) - 1.0
    return r20 - r60


def f_mom_accel_10x60(c, v, o, h, l, m):
    r10 = c / c.shift(10) - 1.0
    r60 = c / c.shift(60) - 1.0
    return r10 - r60


# ---------- (D) drawdown depth ----------
def f_dd_depth_60(c, v, o, h, l, m, win=60):
    return c / c.rolling(win).max() - 1.0


def f_dd_depth_120(c, v, o, h, l, m, win=120):
    return c / c.rolling(win).max() - 1.0


def f_dd_depth_252(c, v, o, h, l, m, win=252):
    return c / c.rolling(win).max() - 1.0


def f_dd_speed_60x120(c, v, o, h, l, m):
    """Drawdown speed: depth from 60d high minus depth from 120d high
    (positive = recovered from deeper drawdown)."""
    d60 = c / c.rolling(60).max() - 1.0
    d120 = c / c.rolling(120).max() - 1.0
    return d60 - d120


# ---------- (E) tradable-anchored conditional betas ----------
def _m(c, m, key):
    return m[key].reindex(c.index).ffill()


def _beta(asset_ret, anchor_ret, win):
    cov = asset_ret.rolling(win).cov(anchor_ret)
    var = anchor_ret.rolling(win).var()
    return cov / var


def make_anchor_cond(key, win=60, look=20):
    def fn(c, v, o, h, l, m, win=win, look=look):
        a = _m(c, m, key)
        ar = a.pct_change()
        br = _beta(c.pct_change(), ar, win)
        move = a / a.shift(look) - 1.0
        return br * move
    fn.__name__ = f"{key.lower()}_beta_cond_{win}x{look}"
    return fn


# ---------- (F) vol-adjusted momentum ----------
def f_mom_vol_adj_20(c, v, o, h, l, m, win=20):
    r = c.pct_change()
    rv = r.rolling(win).std()
    return (c / c.shift(win) - 1.0) / rv


def f_mom_vol_adj_60(c, v, o, h, l, m, win=60):
    r = c.pct_change()
    rv = r.rolling(win).std()
    return (c / c.shift(win) - 1.0) / rv


# ---------- (G) range compression / volume trend ----------
def f_range_ratio_10x60(c, v, o, h, l, m):
    rng = (h - l) / c
    return rng.rolling(10).mean() / rng.rolling(60).mean()


def f_vol_ratio_20x60(c, v, o, h, l, m):
    vv = v.replace(0, np.nan)
    return vv.rolling(20).mean() / vv.rolling(60).mean()


def f_vol_z_60(c, v, o, h, l, m, win=60):
    vv = np.log(v.replace(0, np.nan))
    mu = vv.rolling(win).mean()
    sd = vv.rolling(win).std()
    return (vv - mu) / sd


cands = [
    # (A) intraday price structure
    ("intraday_pos_20", f_intraday_pos_20, "20d mean close location within day range"),
    ("body_ratio_20", f_body_ratio_20, "20d mean |body|/range"),
    ("upper_shadow_20", f_upper_shadow_20, "20d mean upper shadow / range"),
    ("lower_shadow_20", f_lower_shadow_20, "20d mean lower shadow / range"),
    # (B) overnight gaps
    ("overnight_gap_20", f_overnight_gap_20, "20d mean overnight gap (open vs prev close)"),
    ("overnight_gap_60", f_overnight_gap_60, "60d mean overnight gap"),
    ("overnight_gap_abs_20", f_overnight_gap_vol_20, "20d mean |overnight gap|"),
    # (C) momentum acceleration
    ("mom_accel_10x20", f_mom_accel_10x20, "10d ret - 20d ret"),
    ("mom_accel_20x60", f_mom_accel_20x60, "20d ret - 60d ret"),
    ("mom_accel_10x60", f_mom_accel_10x60, "10d ret - 60d ret"),
    # (D) drawdown depth
    ("dd_depth_60", f_dd_depth_60, "close/60d high - 1"),
    ("dd_depth_120", f_dd_depth_120, "close/120d high - 1"),
    ("dd_depth_252", f_dd_depth_252, "close/252d high - 1"),
    ("dd_speed_60x120", f_dd_speed_60x120, "60d dd depth - 120d dd depth"),
    # (E) tradable-anchored conditional betas
    ("spx_beta_cond_60x20", make_anchor_cond("SPX"), "60d beta to SPX x 20d SPX move"),
    ("xau_beta_cond_60x20", make_anchor_cond("XAU"), "60d beta to XAU x 20d XAU move"),
    ("btc_beta_cond_60x20", make_anchor_cond("BTC"), "60d beta to BTC x 20d BTC move"),
    ("wti_beta_cond_60x20", make_anchor_cond("WTI"), "60d beta to WTI x 20d WTI move"),
    # (F) vol-adjusted momentum
    ("mom_vol_adj_20", f_mom_vol_adj_20, "20d ret / 20d vol"),
    ("mom_vol_adj_60", f_mom_vol_adj_60, "60d ret / 60d vol"),
    # (G) range compression / volume trend
    ("range_ratio_10x60", f_range_ratio_10x60, "10d/60d mean daily range ratio"),
    ("vol_ratio_20x60", f_vol_ratio_20x60, "20d/60d mean volume ratio"),
    ("vol_z_60", f_vol_z_60, "log-vol z-score vs 60d"),
]

results = {}
for name, fn, desc in cands:
    res = validate_factor(fn, close, vol, open_, high, low, macro)
    res["max_abs_library_correlation"] = round(max_library_corr(res["panel"], lib), 4)
    results[name] = res
    print_result(f"{name} [{desc}]", res)
    print(f"  max_abs_library_correlation: {res['max_abs_library_correlation']}")

print("\n===== SUMMARY (gate: |IC|>=%.4f, |ICIR|>=%.4f, libcorr<0.5) =====" % (IC_GATE, ICIR_GATE))
for name, res in sorted(results.items()):
    ok = abs(res["ic"]) >= IC_GATE and abs(res["icir"]) >= ICIR_GATE
    libok = res["max_abs_library_correlation"] < 0.5
    flag = "PASS" if (ok and libok) else ("GATE-OK-LIBCORR-HI" if ok else "fail")
    print(f"{name:26s} IC={res['ic']:+.4f} ICIR={res['icir']:+.4f} hit={res['ic_hit_ratio']:.3f} "
          f"cov_ad={res['coverage_asset_days']:.3f} cov8={res['coverage_dates_ge8']:.3f} "
          f"to={res['turnover_10d_rank']:.2f} n={res['n_ic_dates']} libcorr={res['max_abs_library_correlation']:.3f} -> {flag}")
