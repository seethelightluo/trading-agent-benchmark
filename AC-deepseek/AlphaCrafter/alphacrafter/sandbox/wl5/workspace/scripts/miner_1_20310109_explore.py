"""miner_1 2031-01-09: re-validate 2030-12-26 candidates with FIXED rank IC
(pairwise-complete per date, >=8 valid instruments) + screen new candidates.

Regime context (through 2031-01-08): VIX high (41.09 on 12-25, elevated),
ETH +21.4% sharp rebound after crash legs vs BTC -8.6% (extreme crypto
divergence flipped), SOX +14.8% relief bounce after 8 straight down blocks,
WTI +3.6% 7th up in 8 blocks (+144.6% cum), COPPER 5th down block,
SX5E -5.8% soft Europe, XAU haven bid fading (-0.9%), SPX -1.7% US stall.

Candidate ideas:
 [re-validation of 12-26 batch]
 1. eff_ratio_60_signed  - Kaufman efficiency 60d signed
 2. ret_autocorr_20      - lag-1 return autocorrelation 20d
 3. downside_vol_ratio_20- downside semidev / total stdev 20d
 4. yld_beta_60          - rolling beta to US10Y daily change
 5. gain_loss_asym_60    - mean(+ret)/|mean(-ret)| 60d
 6. range_pos_60         - (close-min60)/(max60-min60)
 7. jpy_beta_60          - rolling beta to USDJPY daily change (yen carry proxy)
 8. vol_ratio_10_60      - realized vol 10d / 60d
 [new this cycle]
 9. semi_beta_60         - rolling beta of each asset to SOX returns (semis sensitivity)
10. vol_squeeze_20x60    - (20d high-low range)/(60d high-low range) volatility contraction
11. sharpe_20d           - mean/std of daily returns over 20d (risk-adjusted momentum)
12. mom_rev_5_20         - 5d return - 20d return (short-term reversal vs medium trend)
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from miner_1_20310109_common import (WATCH, MACRO, price_panel, macro_panel, full_validation,
                                     _build_lib_panels,
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


def semi_beta_60(close, ref, window=60, minp=30):
    ra = close.pct_change()
    rb = ref.pct_change()
    cov = ra.rolling(window, min_periods=minp).cov(rb)
    var = rb.rolling(window, min_periods=minp).var()
    return cov.div(var, axis=0).replace([np.inf, -np.inf], np.nan)


def vol_squeeze_20x60(close, w1=20, w2=60, minp1=10, minp2=30):
    lo1 = close.rolling(w1, min_periods=minp1).min()
    hi1 = close.rolling(w1, min_periods=minp1).max()
    lo2 = close.rolling(w2, min_periods=minp2).min()
    hi2 = close.rolling(w2, min_periods=minp2).max()
    return ((hi1 - lo1) / close) / ((hi2 - lo2) / close)


def sharpe_20d(close, window=20, minp=10):
    r = close.pct_change()
    mu = r.rolling(window, min_periods=minp).mean()
    sd = r.rolling(window, min_periods=minp).std()
    return mu / sd


def mom_rev_5_20(close, w_short=5, w_long=20):
    r5 = close / close.shift(w_short) - 1.0
    r20 = close / close.shift(w_long) - 1.0
    return r5 - r20


def main():
    close = price_panel()
    us10y_ret = close["US10Y"] if "US10Y" in close else None
    usdjpy = macro_panel("USDJPY")
    sox = close["SOX"] if "SOX" in close else None

    cands = {
        "eff_ratio_60_signed": eff_ratio_60_signed(close),
        "ret_autocorr_20": ret_autocorr_20(close),
        "downside_vol_ratio_20": downside_vol_ratio_20(close),
        "yld_beta_60": yld_beta_60(close, us10y_ret),
        "gain_loss_asym_60": gain_loss_asym_60(close),
        "range_pos_60": range_pos_60(close),
        "jpy_beta_60": jpy_beta_60(close, usdjpy),
        "vol_ratio_10_60": vol_ratio_10_60(close),
        "semi_beta_60": semi_beta_60(close, sox),
        "vol_squeeze_20x60": vol_squeeze_20x60(close),
        "sharpe_20d": sharpe_20d(close),
        "mom_rev_5_20": mom_rev_5_20(close),
    }

    macro = {m: macro_panel(m) for m in MACRO}
    lib_panels = _build_lib_panels(close, macro)
    rows = []
    for k, (name, panel) in enumerate(cands.items()):
        print(f"validating {name} ({k+1}/{len(cands)})...", flush=True)
        res = full_validation(panel, close, macro, label=name, horizon=10, lib_panels=lib_panels)
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
            "cov_dates_ge8": round(m["coverage_dates_ge8"], 3),
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

    with open("scripts/miner_1_20310109_explore_results.json", "w") as f:
        json.dump(rows, f, indent=1, default=str)


if __name__ == "__main__":
    main()
