"""miner_2 2026-07-30: explore RETURN DISTRIBUTION SHAPE family (skew/kurt/streak/high-age).
Motivation: crash-risk (skew), tail-shape (kurt), trend persistence (streak), cycle aging
(days since 252d high) are classic cross-asset timing signals not yet covered by the library."""
from __future__ import annotations
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_2_lib import validate_factor, load_panel, load_macro, per_asset


@per_asset
def skew_60(s: pd.Series) -> pd.Series:
    return s.pct_change().rolling(60).skew()


@per_asset
def kurt_60(s: pd.Series) -> pd.Series:
    return s.pct_change().rolling(60).kurt()


@per_asset
def streak_10(s: pd.Series) -> pd.Series:
    """Signed current streak: +n if n consecutive up days, -n if n consecutive down days."""
    r = s.pct_change()
    out = pd.Series(np.nan, index=s.index)
    sign = np.sign(r.fillna(0.0))
    run = 0
    last = 0.0
    vals = []
    for v in sign.values:
        if v == 0:
            vals.append(0.0)
            run = 0
            last = 0.0
        elif v == last:
            run += 1
            vals.append(run * v)
        else:
            run = 1
            vals.append(run * v)
            last = v
    out[:] = vals
    return out


@per_asset
def high_age_252(s: pd.Series) -> pd.Series:
    """Days since the 252d rolling high was set (0 = at/near high)."""
    roll_max = s.rolling(252, min_periods=60).max()
    age = pd.Series(np.nan, index=s.index)
    # efficient: for each date, find last date where rolling max was touched
    hits = (s >= roll_max).astype(int)
    last_hit = hits.cumsum()
    # last_hit increments at high dates; age = number of days since last hit
    days_since = pd.Series(np.nan, index=s.index)
    counter = 0
    vals = []
    for h in hits.values:
        if h == 1:
            counter = 0
        else:
            counter += 1
        vals.append(counter)
    days_since[:] = vals
    return days_since.where(roll_max.notna())


def main():
    panel = load_panel()
    macro = load_macro()
    results = {}
    for name, fn, dflt_dir in [("skew_60", skew_60, -1.0),
                               ("kurt_60", kurt_60, -1.0),
                               ("streak_10", streak_10, -1.0),
                               ("high_age_252", high_age_252, 1.0)]:
        r = validate_factor(name, fn, direction_override=dflt_dir)
        results[name] = {k: r[k] for k in ("ic_h10", "icir_h10", "hit_h10", "n_dates_h10",
                                           "coverage_asset_days", "coverage_dates_ge8",
                                           "turnover_10d_rank", "max_abs_library_correlation",
                                           "admission_gate", "direction", "raw_ic_h10",
                                           "decay_ic_by_horizon", "library_corrs")}
    json.dump(results, open("scripts/miner_2_cycle5_shape_results.json", "w"), indent=1, default=str)
    print("SAVED scripts/miner_2_cycle5_shape_results.json")


if __name__ == "__main__":
    main()
