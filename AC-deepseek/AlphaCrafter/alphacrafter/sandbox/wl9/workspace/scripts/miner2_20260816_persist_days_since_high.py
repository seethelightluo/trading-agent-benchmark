"""miner_3 persistence script (2026-08-16): persist passing candidate from sweepG.

Candidate that passed the admission gate (abs(IC)>=0.0070 & abs(ICIR)>=0.0840
at h=10 on the 15-asset universe) AND has max_abs_library_correlation < 0.5:
  - days_since_high_60  (IC -0.0319, ICIR -0.1019, max_corr 0.2577)

Recomputes metrics on the warm-up validation window (2020-01-01..2026-07-15),
builds full-grid signal artifacts (library convention), writes
factors/days_since_high_60.json, then verifies by reading back.
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


def days_since_high_60(close, n=60):
    """Days since last at rolling (n-day) high -- pullback age / lag signal."""
    rollmax = close.rolling(n, min_periods=n).max()
    below = (close < rollmax).astype(float)
    # days since last at-high: use expanding count of consecutive below-high days
    out = pd.Series(np.nan, index=close.index)
    streak = 0
    vals = below.values
    for i in range(len(vals)):
        if np.isnan(vals[i]):
            streak = 0
            out.iloc[i] = np.nan
        elif vals[i] == 0:
            streak = 0
            out.iloc[i] = 0.0
        else:
            streak += 1
            out.iloc[i] = float(streak)
    return out


def main():
    closes = load_closes()
    ohlc = load_full_ohlc()
    print("assets loaded:", len(closes), "full ohlc:", len(ohlc))

    fid = "days_since_high_60"
    spec = {
        "factor_name": "Days Since 60d High",
        "expression": "consecutive days close < rolling_max(close,60) (0 if at high)",
        "description": (
            "Pullback age / lag: count of consecutive trading days below the trailing "
            "60-day rolling maximum. High value = prolonged drawdown/pullback from a "
            "recent high. Negative rank IC at 10d horizon indicates that assets further "
            "from their highs (deeper pullbacks) tend to underperform -- a continuation/"
            "trend-structure signal. Cross-asset predictive on the 15-asset universe."
        ),
        "dependencies": ["close"],
        "parameters": {"window": 60},
        "tags": ["price-structure", "pullback", "cross-asset"],
        "vals_fn": lambda a: days_since_high_60(closes[a], 60),
    }

    vals = {a: spec["vals_fn"](a) for a in closes}
    res = evaluate(closes, vals, fid, horizon=10)

    if not res["passed"]:
        print(f"SKIP {fid}: gate fail")
        return
    if res["max_abs_library_correlation"] >= 0.5:
        print(f"SKIP {fid}: lib corr {res['max_abs_library_correlation']:.3f} >= 0.5")
        return

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
        "expected_direction": -1,
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
