"""miner_2 2026-07-30: explore VOLUME-PARTICIPATION / TREND-QUALITY family.
Motivation: volume trend (participation shift), volume spike, illiquidity trend, and
OLS trend t-stat (slope significance) capture flow/conviction signals absent from library."""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_2_lib import validate_factor, load_panel, load_macro, per_asset


@per_asset
def vol_trend_20x60(s: pd.Series) -> pd.Series:
    v = s.rolling(20).mean() / s.rolling(60).mean()
    return v


@per_asset
def vol_spike_5x60(s: pd.Series) -> pd.Series:
    return s.rolling(5).mean() / s.rolling(60).mean()


@per_asset
def amihud_trend_20x60(s: pd.Series) -> pd.Series:
    am = (s.pct_change().abs() / s.rolling(20).mean()).clip(upper=1e6)
    return am.rolling(20).mean() / am.rolling(60).mean()


@per_asset
def slope_tstat_60(s: pd.Series) -> pd.Series:
    """OLS slope of log-price over 60d divided by its standard error (trend conviction)."""
    x = np.arange(60, dtype=float)
    x = x - x.mean()
    out = pd.Series(np.nan, index=s.index)
    lp = np.log(s)
    for i in range(59, len(lp)):
        y = lp.iloc[i - 59:i + 1].values
        if not np.all(np.isfinite(y)):
            continue
        b = np.polyfit(x, y, 1)[0]
        resid = y - (b * x + y.mean())
        sse = np.sum(resid ** 2) / (len(y) - 2)
        se = np.sqrt(sse / np.sum(x ** 2)) if sse > 0 else np.nan
        out.iloc[i] = b / se if se and np.isfinite(se) and se > 0 else np.nan
    return out


def main():
    panel = load_panel()
    macro = load_macro()
    results = {}
    for name, fn, dflt_dir in [("vol_trend_20x60", vol_trend_20x60, 1.0),
                               ("vol_spike_5x60", vol_spike_5x60, -1.0),
                               ("amihud_trend_20x60", amihud_trend_20x60, 1.0),
                               ("slope_tstat_60", slope_tstat_60, 1.0)]:
        r = validate_factor(name, fn, direction_override=dflt_dir)
        results[name] = {k: r[k] for k in ("ic_h10", "icir_h10", "hit_h10", "n_dates_h10",
                                           "coverage_asset_days", "coverage_dates_ge8",
                                           "turnover_10d_rank", "max_abs_library_correlation",
                                           "admission_gate", "direction", "raw_ic_h10",
                                           "decay_ic_by_horizon", "library_corrs")}
    json.dump(results, open("scripts/miner_2_cycle5_vol_results.json", "w"), indent=1, default=str)
    print("SAVED scripts/miner_2_cycle5_vol_results.json")


if __name__ == "__main__":
    main()
