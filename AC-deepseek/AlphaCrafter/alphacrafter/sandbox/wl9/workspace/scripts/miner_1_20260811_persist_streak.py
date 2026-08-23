"""miner_1 persistence (2026-08-11): persist streak_len_14.

Candidate from sweep L - winning-streak persistence (up-day run-length weighted).
Passes admission gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10; and
max_abs_library_correlation=0.4441 < 0.5 (orthogonal to current library).

Builds full-grid signal artifact (library convention), writes factors/streak_len_14.json,
verifies by reading back.
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

LAST_VALIDATED = "2026-08-11"


def load_full_ohlc():
    out = {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        out[a] = df.set_index("date")
    return out


def full_grid_frame(ohlc, values):
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
    return {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape {frame.shape}",
        "columns": list(frame.columns),
        "shape": [int(frame.shape[0]), int(frame.shape[1])],
        "n_valid_values": int(frame.notna().sum().sum()),
        "sha256": hashlib.sha256(data.encode()).hexdigest()[:16],
        "data": data,
    }


def streak_len(close, n=14):
    """Winning-streak persistence: longest run of up-days within trailing n days."""
    up = (close.diff() > 0).astype(float)

    def streak_block(x):
        best, cur = 0, 0
        for v in x:
            cur = cur + 1 if v > 0.5 else 0
            best = max(best, cur)
        return best
    return up.rolling(n, min_periods=7).apply(lambda x: streak_block(x.values), raw=False)


def persist():
    closes = load_closes()
    ohlc = load_full_ohlc()
    fid = "streak_len_14"
    vals = {a: streak_len(closes[a], 14) for a in closes}

    res = evaluate(closes, vals, fid, horizon=10)
    print(f"evaluated {fid}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
          f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}")

    assert res["passed"], "gate fail"
    assert res["max_abs_library_correlation"] < 0.5, f"lib corr {res['max_abs_library_correlation']:.3f} >= 0.5"

    frame = full_grid_frame(ohlc, vals)
    artifact = make_artifact(frame)

    doc = {
        "factor_id": fid,
        "factor_name": "Winning Streak Persistence 14d",
        "version": "1.0.0",
        "calculation": {
            "expression": "longest run of close.diff()>0 within trailing 14d",
            "description": ("Number of consecutive up-days (longest run) in the trailing 14-day "
                            "window. High values indicate persistent, unbroken positive momentum; "
                            "captures trend consistency beyond raw cumulative return."),
        },
        "dependencies": ["close"],
        "parameters": {"window": 14, "min_periods": 7},
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": LAST_VALIDATED,
            "admission_horizon": 10,
            "regime_notes": (
                "Validated 2020-01-01..2026-07-15 across multiple regimes (COVID crash 2020, "
                "2020-21 recovery bull, 2022 tightening bear, 2023-24 AI equity rally, "
                "2024-26 crypto/commodity cycles). Cross-sectional rank IC on the 15-asset "
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
        "tags": ["momentum", "trend", "persistence"],
    }

    path = FACTOR_DIR / f"{fid}.json"
    path.write_text(json.dumps(doc, indent=1))
    print(f"WROTE {path}")

    check = json.loads(path.read_text())
    assert check["factor_id"] == fid, "factor_id mismatch"
    assert check["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert check["validation"]["metrics"]["ic"] == doc["validation"]["metrics"]["ic"]
    assert check["validation"]["metrics"]["icir"] == doc["validation"]["metrics"]["icir"]
    assert check["validation"]["metrics"]["max_abs_library_correlation"] == doc["validation"]["metrics"]["max_abs_library_correlation"]
    assert "data" in check["validation"]["signal_artifact"], "artifact missing"
    print(f"VERIFIED {fid}: json ok, status={check['validation']['status']}, art_shape={check['validation']['signal_artifact']['shape']}")


if __name__ == "__main__":
    persist()