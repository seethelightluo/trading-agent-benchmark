"""miner_3 cycle screening: novel factor families not in current library.
Screens each candidate at h=10 admission gate |IC|>=0.007, |ICIR|>=0.084.
"""
from __future__ import annotations
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import load_panel, load_macro, per_asset, validate_factor


def cand_skew_20(panel, macro):
    return per_asset(lambda s: s.pct_change().rolling(20).skew())(panel, macro)

def cand_skew_60(panel, macro):
    return per_asset(lambda s: s.pct_change().rolling(60).skew())(panel, macro)

def cand_kurt_20(panel, macro):
    return per_asset(lambda s: s.pct_change().rolling(20).kurt())(panel, macro)

def cand_range_vol_ratio_20(panel, macro):
    # Parkinson vol / close-to-close vol over 20d (intraday inefficiency)
    def f(s):
        h = s.rolling(20).max(); l = s.rolling(20).min()
        park = np.sqrt(np.log(h / l) ** 2 / (4 * np.log(2))).rolling(20).mean()
        c2c = s.pct_change().rolling(20).std()
        return park / c2c
    return per_asset(f)(panel, macro)

def cand_drawdown_60(panel, macro):
    # depth below rolling 60d high (negative when below high)
    return per_asset(lambda s: s / s.rolling(60).max() - 1.0)(panel, macro)

def cand_drawdown_120(panel, macro):
    return per_asset(lambda s: s / s.rolling(120).max() - 1.0)(panel, macro)

def cand_vol_z_20(panel, macro):
    # z-score of current 20d vol vs trailing 120d vol distribution
    def f(s):
        v = s.pct_change().rolling(20).std()
        mu = v.rolling(120).mean(); sd = v.rolling(120).std()
        return (v - mu) / sd
    return per_asset(f)(panel, macro)

def cand_vol_trend_ratio(panel, macro):
    # 20d vol / 60d vol (vol contraction => positive)
    return per_asset(lambda s: s.pct_change().rolling(20).std() / s.pct_change().rolling(60).std())(panel, macro)

def cand_volume_trend_20(panel, macro):
    # short-term average volume vs longer-term (volume expansion)
    def f(s):
        vol = s / s.shift(1)  # placeholder, replaced below
        return vol
    return None

def cand_dxy_beta_cond_60x20(panel, macro):
    dxy = macro["DXY"].dropna()
    def f(s):
        r = s.pct_change()
        v = dxy.pct_change().reindex(s.index)
        z = pd.concat([r.rename("r"), v.rename("v")], axis=1).dropna()
        beta = z["r"].rolling(60).cov(z["v"]) / z["v"].rolling(60).var()
        dxy20 = (dxy / dxy.shift(20) - 1.0).reindex(s.index)
        return -beta * dxy20
    return per_asset(f)(panel, macro)

def cand_rev_5d(panel, macro):
    # short-term reversal: negative 5d return (sign-flipped to be positive-alpha)
    return per_asset(lambda s: -(s.shift(5) / s.shift(10) - 1.0))(panel, macro)

def cand_gap_ratio_20(panel, macro):
    # overnight gap intensity: |open - prev close| / close, 20d mean
    def f(s):
        oc = (s / s.shift(1) - 1.0).abs()
        return oc.rolling(20).mean()
    return per_asset(f)(panel, macro)

def cand_amihud_change_20(panel, macro):
    # liquidity improvement: -1 * change in 20d Amihud illiquidity
    def f(s):
        r = s.pct_change().abs()
        ami = r / s  # illiquidity proxy (return per price unit; volume unavailable)
        return -(ami.rolling(20).mean() / ami.rolling(80).mean() - 1.0)
    return per_asset(f)(panel, macro)


if __name__ == "__main__":
    cands = {
        "skew_20d": cand_skew_20,
        "skew_60d": cand_skew_60,
        "kurt_20d": cand_kurt_20,
        "range_vol_ratio_20": cand_range_vol_ratio_20,
        "drawdown_60d": cand_drawdown_60,
        "drawdown_120d": cand_drawdown_120,
        "vol_z_20d": cand_vol_z_20,
        "vol_trend_ratio": cand_vol_trend_ratio,
        "dxy_beta_cond_60x20": cand_dxy_beta_cond_60x20,
        "rev_5d": cand_rev_5d,
        "gap_ratio_20": cand_gap_ratio_20,
        "amihud_change_20": cand_amihud_change_20,
    }
    summary = []
    for name, fn in cands.items():
        try:
            res = validate_factor(name, fn, horizons=(5, 10, 20), print_extra="")
            summary.append((name, res["ic_h10"], res["icir_h10"], res.get("max_abs_library_correlation", float("nan"))))
        except Exception as e:
            print(f"{name}: ERROR {e}")
            summary.append((name, float("nan"), float("nan"), float("nan")))
    print("\n===== SCREEN SUMMARY (h=10) =====")
    for name, ic, icir, mc in summary:
        flag = "PASS" if (abs(ic) >= 0.007 and abs(icir) >= 0.084) else "fail"
        print(f"{name:24s} IC={ic:+.4f} ICIR={icir:+.4f} maxcorr={mc:.3f} -> {flag}")
