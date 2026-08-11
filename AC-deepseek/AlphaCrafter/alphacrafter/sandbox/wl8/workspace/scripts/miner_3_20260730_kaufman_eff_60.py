"""miner_3 2026-07-30 candidate 1: kaufman_eff_60
Efficiency Ratio (Kaufman): |close_t - close_{t-60}| / sum(|close_i - close_{i-1}|, 60).
High ER => clean directional trend; low ER => choppy/rangebound. Hypothesis: in a
cross-asset universe with persistent trends (crypto, commodities, momentum indices),
cleaner trends continue -> positive IC.
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
from miner_3_20260730_common import load_data, run_validation

N = 60


def kaufman_eff(close, n=N):
    direction = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n).sum()
    er = direction / path.replace(0, np.nan)
    return er


def main():
    data = load_data()
    factor = {a: kaufman_eff(d["close"].astype(float)) for a, d in data.items()}
    expr = "abs(close - close.shift(60)) / sum(abs(close.diff()), 60)"
    desc = ("Kaufman efficiency ratio over 60 days: net 60-day move divided by total "
            "absolute path length. High values indicate clean directional trends; low "
            "values indicate choppy rangebound action.")
    out = run_validation(
        factor_id="kaufman_eff_60",
        factor_name="Kaufman Efficiency Ratio 60d",
        expression=expr,
        description=desc,
        deps=["close"],
        params={"lookback": 60},
        factor_series=factor,
        data=data,
        tags=["trend", "trend_quality", "efficiency"],
        regime_notes="Validated 2020-01-01..2026-07-29 on the 15-asset cross-asset "
                     "universe (equity indices, commodities, crypto, yields). Cross-"
                     "sectional rank IC vs 10d forward returns.",
    )
    if out is None:
        return
    metrics, record = out
    from miner_3_20260730_common import persist_factor
    persist_factor(record, factor)


if __name__ == "__main__":
    main()
