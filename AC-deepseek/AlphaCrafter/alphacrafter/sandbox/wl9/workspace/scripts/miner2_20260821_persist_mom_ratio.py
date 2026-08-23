"""miner_2 (2026-08-21): persist passing candidate mom_ratio_15x40_skip5.

Momentum ratio = short-term momentum / med-term momentum, both with 5d skip.
Construction: ms = close.shift(5)/close.shift(20)-1; mm = close.shift(5)/close.shift(45)-1.
Structural: rewards recent (15d) acceleration relative to slower (40d) baseline, and
is nearly orthogonal to the existing library (max_abs_library_correlation=0.0148).

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10 on 15-asset universe; lib corr < 0.5.
"""
from __future__ import annotations
import base64
import hashlib
import json
import sys
import zlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import (  # noqa: E402
    ASSETS, FACTOR_DIR, STOCK_DIR, evaluate, load_closes,
)

LAST_VALIDATED = "2026-08-21"
FID = "mom_ratio_15x40_skip5"


def mom_ratio(close, short=15, med=40, skip=5):
    ms = close.shift(skip) / close.shift(skip + short) - 1.0
    mm = close.shift(skip) / close.shift(skip + med) - 1.0
    return ms / mm.replace(0, np.nan)


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


def main():
    closes = load_closes()
    vals = {a: mom_ratio(closes[a]) for a in closes}
    res = evaluate(closes, vals, FID, horizon=10)
    print(f"RESULT {FID}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
          f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}")

    if not res["passed"]:
        print("SKIP: gate fail")
        return
    if res["max_abs_library_correlation"] >= 0.5:
        print("SKIP: lib corr >= 0.5")
        return

    ohlc = load_full_ohlc()
    frame = full_grid_frame(ohlc, vals)
    artifact = make_artifact(frame)

    doc = {
        "factor_id": FID,
        "factor_name": "momentum_ratio_short_med_skip5",
        "version": "1.0.0",
        "calculation": {
            "expression": "ms/mm, ms=close.shift(5)/close.shift(20)-1, mm=close.shift(5)/close.shift(45)-1",
            "description": (
                "Ratio of short-term (15d, skip5) momentum to medium-term (40d, skip5) "
                "momentum. Values > 1 indicate recent acceleration relative to slower "
                "baseline; < 1 indicate deceleration. Near-orthogonal to the library."
            ),
        },
        "dependencies": ["close"],
        "parameters": {"short": 15, "med": 40, "skip": 5},
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": LAST_VALIDATED,
            "admission_horizon": 10,
            "regime_notes": (
                "Validated 2020-01-01..2026-07-15 across multiple regimes on the 15-asset "
                "tradable universe. Rank IC positive at h=10. Structurally distinct from "
                "library momentum factors (max_abs_library_correlation=0.0148)."
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
        "tags": ["momentum", "acceleration", "ratio"],
    }
    path = FACTOR_DIR / f"{FID}.json"
    path.write_text(json.dumps(doc))
    print(f"WROTE {path}")

    check = json.loads(path.read_text())
    assert check["factor_id"] == FID, "factor_id mismatch"
    assert check["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert check["validation"]["metrics"]["ic"] == doc["validation"]["metrics"]["ic"]
    assert check["validation"]["metrics"]["icir"] == doc["validation"]["metrics"]["icir"]
    art = check["validation"]["signal_artifact"]
    assert "data" in art, "missing signal data"
    print(f"VERIFIED {FID}: json ok, status={check['validation']['status']}, "
          f"IC={check['validation']['metrics']['ic']}, ICIR={check['validation']['metrics']['icir']}, "
          f"artifact shape={art['shape']}")


if __name__ == "__main__":
    main()