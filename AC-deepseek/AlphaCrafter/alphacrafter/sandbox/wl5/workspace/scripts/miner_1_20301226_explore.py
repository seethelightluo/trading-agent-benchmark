"""miner_1 2030-12-26 exploration batch: screen 8 orthogonal candidate factor ideas.

Regime context (through 2030-12-25): VIX high (37.8 on 11-27, still elevated), crypto
crash legs (BTC -9.8%, ETH -17.1% then divergence BTC +13.5% vs ETH -7.1%), SOX 8th
straight down block, NDX sharp reversal (-14% after +7.6%), WTI strong uptrend (+144.6%
cum), XAU haven bid, COPPER 4th down block, FX flat (DXY ~97.8, USDCNY 6.73).

Candidate ideas:
 1. eff_ratio_60_signed  - Kaufman efficiency: (P_t - P_{t-60}) / sum(|dP|) over 60d, signed trend efficiency
 2. ret_autocorr_20      - lag-1 autocorrelation of daily returns over 20d (return persistence / reversal)
 3. downside_vol_ratio_20- downside semideviation / total stdev over 20d (downside risk share)
 4. yld_beta_60          - rolling beta of asset returns to US10Y daily change (rates sensitivity)
 5. gain_loss_asym_60    - mean(+ret)/|mean(-ret)| over 60d (upside/downside capture asymmetry)
 6. range_pos_60         - (close - min60)/(max60 - min60) position in 60d range
 7. jpy_beta_60          - rolling beta of asset returns to USDJPY daily change (yen carry proxy)
 8. vol_ratio_10_60      - realized vol 10d / 60d (per-asset vol term structure)
"""
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from miner_1_20301226_common import (WATCH, MACRO, price_panel, macro_panel, full_validation,
                                     IC_THRESHOLD, ICIR_THRESHOLD, CORR_THRESHOLD, VISIBLE_THROUGH)


def eff_ratio_60_signed(close, window=60):
    path = close.pct_change().abs().rolling(window, min_periods=30).sum()
    net = close - close.shift(window)
    return (net / path).replace([np.inf, -np.inf], np.nan)


def ret_autocorr_20(close, window=20, minp=10):
    r = close.pct_change()
    return r.rolling(window, min_periods=minp).apply(
        lambda x: pd.Series(x).autocorr(lag=1) if len(x) >= minp else np.nan, raw=False)


def downside_vol_ratio_20(close, window=20, minp=10):
    r = close.pct_change()
    neg = r.clip(upper=0)
    dv = neg.rolling(window, min_periods=minp).std()
    tv = r.rolling(window, min_periods=minp).std()
    return dv / tv


def yld_beta_60(close, us10y, window=60, minp=30):
    ra = close.pct_change()
    rb = us10y.pct_change()
    cov = ra.rolling(window, min_periods=minp).cov(rb)
    var = rb.rolling(window, min_periods=minp).var()
    return cov.div(var, axis=0).replace([np.inf, -np.inf], np.nan)


def gain_loss_asym_60(close, window=60, minp=30):
    r = close.pct_change()
    up = r.where(r > 0)
    dn = r.where(r < 0)
    upm = up.rolling(window, min_periods=minp).mean()
    dnm = dn.rolling(window, min_periods=minp).mean().abs()
    return upm / dnm


def range_pos_60(close, window=60, minp=30):
    lo = close.rolling(window, min_periods=minp).min()
    hi = close.rolling(window, min_periods=minp).max()
    return (close - lo) / (hi - lo)


def jpy_beta_60(close, usdjpy, window=60, minp=30):
    ra = close.pct_change()
    rb = usdjpy.pct_change()
    cov = ra.rolling(window, min_periods=minp).cov(rb)
    var = rb.rolling(window, min_periods=minp).var()
    return cov.div(var, axis=0).replace([np.inf, -np.inf], np.nan)


def vol_ratio_10_60(close, w1=10, w2=60, minp=5):
    v1 = close.pct_change().rolling(w1, min_periods=minp).std()
    v2 = close.pct_change().rolling(w2, min_periods=30).std()
    return v1 / v2


def main():
    close = price_panel()
    us10y = macro_panel("US10Y") if False else None  # US10Y is a watch asset itself
    us10y_ret = close["US10Y"] if "US10Y" in close else None
    usdjpy = macro_panel("USDJPY")

    cands = {
        "eff_ratio_60_signed": eff_ratio_60_signed(close),
        "ret_autocorr_20": ret_autocorr_20(close),
        "downside_vol_ratio_20": downside_vol_ratio_20(close),
        "yld_beta_60": yld_beta_60(close, us10y_ret),
        "gain_loss_asym_60": gain_loss_asym_60(close),
        "range_pos_60": range_pos_60(close),
        "jpy_beta_60": jpy_beta_60(close, usdjpy),
        "vol_ratio_10_60": vol_ratio_10_60(close),
    }

    macro = {m: macro_panel(m) for m in MACRO}
    rows = []
    for name, panel in cands.items():
        res = full_validation(panel, close, macro, label=name, horizon=10)
        m = res["metrics"]
        gate_ic = abs(m["ic"]) >= IC_THRESHOLD
        gate_icir = abs(m["icir"]) >= ICIR_THRESHOLD
        gate_corr = abs(m["max_abs_library_correlation"]) <= CORR_THRESHOLD
        rows.append({
            "factor": name,
            "ic10": round(m["ic"], 5),
            "icir10": round(m["icir"], 4),
            "hit": round(m["ic_hit_ratio"], 3),
            "n_dates": m["n_ic_dates"],
            "cov_asset_days": round(m["coverage_asset_days"], 3),
            "turnover": round(m["turnover_10d_rank"], 3),
            "max_abs_lib_corr": round(m["max_abs_library_correlation"], 3),
            "pass_ic": gate_ic, "pass_icir": gate_icir, "pass_corr": gate_corr,
            "decay": {k: round(v, 4) for k, v in m["decay_ic_by_horizon"].items()},
        })
        print(f"=== {name} ===")
        print(f"  IC10={m['ic']:.5f} ICIR10={m['icir']:.4f} hit={m['ic_hit_ratio']:.3f} n={m['n_ic_dates']}")
        print(f"  coverage_asset_days={m['coverage_asset_days']:.3f} dates_ge8={m['coverage_dates_ge8']:.3f} turnover={m['turnover_10d_rank']:.3f}")
        print(f"  max_abs_lib_corr={m['max_abs_library_correlation']:.3f}")
        print(f"  regime: { {k: (round(v['ic'],4), round(v['icir'],2), v['n']) for k,v in res['regime_split'].items()} }")
        print(f"  decay: { {k: round(v,4) for k,v in m['decay_ic_by_horizon'].items()} }")
        print(f"  gate: IC={gate_ic} ICIR={gate_icir} CORR={gate_corr}")
        print()

    print("SUMMARY TABLE")
    for r in rows:
        print(r)

    import json
    with open("scripts/miner_1_20301226_explore_results.json", "w") as f:
        json.dump(rows, f, indent=1, default=str)


if __name__ == "__main__":
    main()
