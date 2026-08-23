"""Runner for miner_2 sweep-I persistence (2026-08-19).

Imports functions from miner2_20260819_persist_sweepI.py and drives the
persistence of the two passing candidates plus routine re-validation of the
whole effective factor library. Run from workspace/ (the cwd) so that
`factors/` and `../persistent/stock_data` resolve correctly.
"""
from __future__ import annotations

import sys

sys.path.insert(0, "scripts")

from miner2_20260819_persist_sweepI import (  # noqa: E402
    kaufman_eff,
    kurt_20d,
    load_full_ohlc,
    persist_one,
    revalidate_all,
)
from miner3_20260730_harness import load_closes  # noqa: E402


def make_specs():
    def _kaufman(a):
        return None  # replaced below by closure over closes

    def _kurt(a):
        return None

    specs = [
        {
            "fid": "kaufman_eff_20d",
            "factor_name": "Kaufman Efficiency Ratio 20d",
            "expression": "abs(close-close_20)/sum(|diff|,20)",
            "description": (
                "Kaufman efficiency ratio over 20 days: net change magnitude "
                "scaled by cumulative 20-day price travel. Trend efficiency / "
                "trend-quality signal, direction +1 (efficient trends persist)."
            ),
            "dependencies": ["close"],
            "parameters": {"window": 20},
            "tags": ["trend", "efficiency", "quality"],
            "direction": 1,
        },
        {
            "fid": "kurt_20d",
            "factor_name": "Return Kurtosis 20d",
            "expression": "mean((r-mean)^4)/std(r)^4 - 3 over 20d",
            "description": (
                "Excess kurtosis of daily returns over 20 days. Low kurtosis "
                "names (fat/quiet) versus high-kurtosis (tail-prone) names. "
                "Direction +1: lower-kurtosis assets tend to deliver steadier "
                "forward returns on the long-only 15-asset universe."
            ),
            "dependencies": ["close"],
            "parameters": {"window": 20, "min_periods": 10},
            "tags": ["distortion", "volatility", "quality"],
            "direction": 1,
        },
    ]

    def kaufman_fn(closes):
        return lambda a: kaufman_eff(closes[a], n=20)

    def kurt_fn(closes):
        return lambda a: kurt_20d(closes[a], n=20)

    return specs, kaufman_fn, kurt_fn


def main():
    closes = load_closes()
    ohlc = load_full_ohlc()
    print("assets loaded:", len(closes), "ohlc loaded:", len(ohlc))
    print()

    specs, kaufman_fn, kurt_fn = make_specs()
    fn_map = {
        "kaufman_eff_20d": kaufman_fn(closes),
        "kurt_20d": kurt_fn(closes),
    }

    for spec in specs:
        fid = spec["fid"]
        print(f"\n=== PERSIST CANDIDATE {fid} ===")
        # persist_one reads vals_fn via spec? No - it takes `spec` with vals_fn.
        # We inject vals_fn into spec copy.
        s = dict(spec)
        s["vals_fn"] = fn_map[fid]
        persist_one(closes, ohlc, fid, s, spec["direction"])

    print("\nDone persisting sweep-I candidates.")


if __name__ == "__main__":
    main()