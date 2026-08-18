"""miner_3 persistence script (2026-08-16): persist passing candidates from sweepC.

Candidates that passed the admission gate (abs(IC)>=0.0070 & abs(ICIR)>=0.0840
at h=10 on the 15-asset universe) AND have max_abs_library_correlation < 0.5:
  - bb_width_20d      (IC 0.0452, ICIR 0.1380, max_corr 0.3711)
  - hi_lo_range_20d   (IC 0.0340, ICIR 0.1017, max_corr 0.3879)

Recomputes metrics on the warm-up validation window (2020-01-01..2026-07-15),
builds full-grid signal artifacts (library convention), writes
factors/<factor_id>.json, then verifies by reading back.

rsi_14d also passed the IC gate (IC 0.0481, ICIR 0.1474) but is skipped here
because max_abs_library_correlation = 0.853 (pairwise conflict with rng_pos_20d).
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import sys
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import (  # noqa: E402
    ASSETS,
    FACTOR_DIR,
    STOCK_DIR,
    VALID_END,
    evaluate,
    load_closes,
)

LAST_VALIDATED = "2026-08-16"


def load_full_ohlc():
    """Load full close/high/low series (no cap) for artifact building."""
    out = {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        out[a] = df.set_index("date")
    return out


def full_grid_frame(ohlc, values):
    """Align per-asset factor series onto the full weekday grid (no cap)."""
    dates = sorted({d for df in ohlc.values() for d in df.index})
    dates = [d for d in dates if d.weekday() < 5]
    grid = pd.DatetimeIndex(dates)
    df = pd.DataFrame(index=grid, columns=ASSETS, dtype=float)
    for a, s in values.items():
        if a in df.columns:
            df[a] = s.reindex(grid)
    return df


def make_artifact(frame):
    csv = frame.to_csv()
    data = base64.b64encode(zlib.compress(csv.encode())).decode()
    n_valid = int(frame.notna().sum().sum())
    return {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape {frame.shape}",
        "columns": list(frame.columns),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "n_valid_values": n_valid,
        "sha256": hashlib.sha256(data.encode()).hexdigest()[:16],
        "data": data,
    }


def bb_width_20(close, n=20):
    """Bollinger bandwidth: 4*std(close,n)/SMA(close,n) -- volatility level."""
    sma = close.rolling(n).mean()
    sd = close.rolling(n).std(ddof=0)
    return 4.0 * sd / sma.replace(0, np.nan)


def hi_lo_range_20(close, high, low, n=20):
    """(rolling_max(high,n) - rolling_min(low,n)) / close -- range width normalized."""
    hi = high.rolling(n).max()
    lo = low.rolling(n).min()
    return (hi - lo) / close.replace(0, np.nan)


def main():
    closes = load_closes()
    ohlc = load_full_ohlc()
    print("assets loaded:", len(closes), "full ohlc:", len(ohlc))

    highs = {a: ohlc[a]["high"].astype(float) for a in closes}
    lows = {a: ohlc[a]["low"].astype(float) for a in closes}

    candidates = {
        "bb_width_20d": {
            "factor_name": "Bollinger Bandwidth 20d",
            "expression": "4 * rolling_std(close,20) / rolling_mean(close,20)",
            "description": "Bollinger bandwidth: standardized trailing 20-day volatility level. High bandwidth = expanding volatility regime; cross-asset predictive at 10d horizon.",
            "dependencies": ["close"],
            "parameters": {"window": 20},
            "tags": ["volatility", "cross-asset"],
            "vals_fn": lambda a: bb_width_20(closes[a], 20),
        },
        "hi_lo_range_20d": {
            "factor_name": "High-Low Range Width 20d",
            "expression": "(rolling_max(high,20) - rolling_min(low,20)) / close",
            "description": "Normalized width of the trailing 20-day high-low range. Range expansion/volatility-regime signal, close-normalized for cross-asset comparability.",
            "dependencies": ["close", "high", "low"],
            "parameters": {"window": 20},
            "tags": ["volatility", "price-structure", "cross-asset"],
            "vals_fn": lambda a: hi_lo_range_20(closes[a], highs[a], lows[a], 20),
        },
    }

    results = {}
    for fid, spec in candidates.items():
        print(f"\n=== recompute {fid} ===")
        vals = {a: spec["vals_fn"](a) for a in closes}
        res = evaluate(closes, vals, fid, horizon=10)
        results[fid] = (spec, res, vals)

    # Persist PASS candidates with max_abs_library_correlation < 0.5
    for fid, (spec, res, vals) in results.items():
        if not res["passed"]:
            print(f"SKIP {fid}: gate fail")
            continue
        if res["max_abs_library_correlation"] >= 0.5:
            print(f"SKIP {fid}: lib corr {res['max_abs_library_correlation']:.3f} >= 0.5")
            continue
        frame = full_grid_frame(ohlc, vals)
        artifact = make_artifact(frame)

        doc = {
            "factor_id": fid,
            "factor_name": spec["factor_name"],
            "version": "1.0.0",
            "calculation": {
                "expression": spec["expression"],
                "description": spec["description"],
            },
            "dependencies": spec["dependencies"],
            "parameters": spec["parameters"],
            "expected_direction": 1,
            "validation": {
                "status": "EFFECTIVE",
                "period": "2020-01-01..2026-07-15",
                "last_validated": LAST_VALIDATED,
                "admission_horizon": 10,
                "regime_notes": (
                    "Validated 2020-01-01..2026-07-15 across multiple regimes: COVID crash 2020, "
                    "recovery bull 2020-21, 2022 tightening bear, 2023-24 AI-led equity rally, "
                    "2024-26 crypto/commodity cycles. Cross-sectional rank IC on the 15-asset "
                    "tradable universe."
                ),
                "metrics": {
                    "ic": round(res["ic"], 4),
                    "icir": round(res["icir"], 4),
                    "ic_hit_ratio": round(res["hit"], 4),
                    "n_ic_dates": int(res["n_ic_dates"]),
                    "coverage_asset_days": round(res["coverage_asset_days"], 4),
                    "coverage_dates_ge8": round(res["coverage_dates_ge8"], 4),
                    "turnover_10d_rank": round(res["turnover_10d_rank"], 4),
                    "decay_ic_by_horizon": {k: round(float(v), 4) for k, v in res["decay"].items()},
                    "max_abs_library_correlation": round(res["max_abs_library_correlation"], 4),
                },
                "signal_artifact": artifact,
            },
            "tags": spec["tags"],
        }

        path = FACTOR_DIR / f"{fid}.json"
        path.write_text(json.dumps(doc))
        print(f"WROTE {path}")

        # Read back + verify
        check = json.loads(path.read_text())
        assert check["factor_id"] == fid, "factor_id mismatch"
        assert check["validation"]["status"] == "EFFECTIVE", "status mismatch"
        assert check["validation"]["metrics"]["ic"] == doc["validation"]["metrics"]["ic"]
        assert check["validation"]["metrics"]["icir"] == doc["validation"]["metrics"]["icir"]
        assert check["validation"]["metrics"]["max_abs_library_correlation"] == doc["validation"]["metrics"]["max_abs_library_correlation"]
        assert "data" in check["validation"]["signal_artifact"], "artifact missing"
        print(f"VERIFIED {fid}: json ok, status={check['validation']['status']}, "
              f"IC={check['validation']['metrics']['ic']}, ICIR={check['validation']['metrics']['icir']}, "
              f"artifact shape={check['validation']['signal_artifact']['shape']}")

    print("\nDONE")


if __name__ == "__main__":
    main()