"""miner_3 (2026-08-16): persistence of days_since_high_60.

Sweep G found this candidate passed the admission gate
(abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10) with IC -0.0319 / ICIR -0.1019
and max_abs_library_correlation 0.2577 < 0.5 (distinct from the momentum /
price-structure / vol family already in the library).

Construction: number of consecutive trading days (0-based) since close last
made a new trailing 60-day high. High value = asset has been persistently
below its recent high (weak/struggling trend). Negative IC at h=10 => high
days-since-high predicts LOWER forward 10-day returns (trend-persistence /
lack of breakout). Direction -1.
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
from miner3_20260730_harness import FACTOR_DIR, STOCK_DIR, evaluate, load_closes  # noqa: E402

LAST_VALIDATED = "2026-08-16"
FID = "days_since_high_60"


def load_full_ohlc():
    out = {}
    for f in sorted(STOCK_DIR.glob("*.csv")):
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        out[f.stem] = df.set_index("date")
    return out


def full_grid_frame(ohlc, values):
    dates = sorted({d for df in ohlc.values() for d in df.index})
    dates = [d for d in dates if d.weekday() < 5]
    grid = pd.DatetimeIndex(dates)
    cols = list(values.keys())
    df = pd.DataFrame(index=grid, columns=cols, dtype=float)
    for a, s in values.items():
        df[a] = s.reindex(grid)
    return df


def make_artifact(frame):
    csv = frame.to_csv()
    data = base64.b64encode(zlib.compress(csv.encode())).decode()
    return {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape {frame.shape}",
        "columns": list(frame.columns),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "n_valid_values": int(frame.notna().sum().sum()),
        "sha256": hashlib.sha256(data.encode()).hexdigest()[:16],
        "data": data,
    }


def days_since_high_60(close, n=60):
    rollmax = close.rolling(n, min_periods=n).max()
    below = (close < rollmax).astype(float)
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

    vals = {a: days_since_high_60(closes[a], 60) for a in closes}
    res = evaluate(closes, vals, FID, horizon=10)
    if not res["passed"]:
        print(f"GATE FAIL for {FID}, not persisting")
        return
    if res["max_abs_library_correlation"] >= 0.5:
        print(f"SKIP {FID}: lib corr {res['max_abs_library_correlation']:.3f} >= 0.5")
        return

    frame = full_grid_frame(ohlc, vals)
    artifact = make_artifact(frame)
    doc = {
        "factor_id": FID,
        "factor_name": "Days Since High 60d",
        "version": "1.0.0",
        "calculation": {
            "expression": "consecutive count of trading days since close last >= rolling_max(close,60)",
            "description": ("Number of consecutive trading days (0-based) that close has remained below "
                            "its trailing 60-day high. High value = persistently weak/struggling trend. "
                            "Direction -1: high days-since-high predicts lower forward 10-day returns."),
        },
        "dependencies": ["close"],
        "parameters": {"window": 60},
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
        "tags": ["price-structure", "trend", "drawdown-age", "cross-asset"],
    }

    path = FACTOR_DIR / f"{FID}.json"
    path.write_text(json.dumps(doc))
    print(f"WROTE {path}")

    check = json.loads(path.read_text())
    assert check["factor_id"] == FID
    assert check["validation"]["status"] == "EFFECTIVE"
    assert check["validation"]["metrics"]["ic"] == doc["validation"]["metrics"]["ic"]
    assert check["validation"]["metrics"]["icir"] == doc["validation"]["metrics"]["icir"]
    assert check["validation"]["metrics"]["max_abs_library_correlation"] == doc["validation"]["metrics"]["max_abs_library_correlation"]
    assert check["validation"]["metrics"]["max_abs_library_correlation"] < 0.5
    assert check["validation"]["signal_artifact"]["n_valid_values"] > 0
    print("READ-BACK VERIFY OK:", path)


if __name__ == "__main__":
    main()
