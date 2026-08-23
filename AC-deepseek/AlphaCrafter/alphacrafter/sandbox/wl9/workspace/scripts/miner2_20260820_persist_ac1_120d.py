"""miner_2 (2026-08-20): persist passing candidate ac1_120d from sweep N.

ac1_120d = lag-1 autocorrelation of daily returns over a 120-day window,
representing long-horizon trend memory / return persistence. Negative signal:
assets whose recent daily returns display strong negative lag-1 autocorrelation
(regime separation/oscillation) tend to underperform over the 10-day forward
horizon (negative IC -> direction -1).

Gate: abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10 on 15-asset universe.
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
    ASSETS,
    FACTOR_DIR,
    STOCK_DIR,
    evaluate,
    load_closes,
)

LAST_VALIDATED = "2026-08-20"
FID = "ac1_120d"


def autocorr_lag1(close, n=120):
    r = close.pct_change()
    r0 = r.rolling(n, min_periods=n // 2).mean()
    r1 = r.shift(1)
    cov = ((r - r0) * (r1 - r0)).rolling(n, min_periods=n // 2).mean()
    var = ((r - r0) ** 2).rolling(n, min_periods=n // 2).mean().replace(0, np.nan)
    return cov / var


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
    vals = {a: autocorr_lag1(closes[a], 120) for a in closes}
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
        "factor_name": "return_autocorr_lag1_120d",
        "version": "1.0.0",
        "calculation": {
            "expression": "cov(r, r.shift(1)) / var(r) over rolling 120d, r = close.pct_change()",
            "description": (
                "Lag-1 autocorrelation of daily returns over a 120-day window. "
                "Persistence/memory of return direction at a long horizon; "
                "negative values indicate strong return mean-reversion."
            ),
        },
        "dependencies": ["close"],
        "parameters": {"window": 120, "min_periods": 60},
        "expected_direction": -1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": LAST_VALIDATED,
            "admission_horizon": 10,
            "regime_notes": (
                "Validated 2020-01-01..2026-07-15 across multiple regimes. "
                "Cross-sectional rank IC on the 15-asset tradable universe. "
                "Negative direction: strong long-window return mean-reversion "
                "signals relative underperformance over 10d."
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
        "tags": ["autocorrelation", "mean_reversion", "risk"],
    }

    path = FACTOR_DIR / f"{FID}.json"
    path.write_text(json.dumps(doc))
    print(f"WROTE {path}")

    check = json.loads(path.read_text())
    assert check["factor_id"] == FID, "factor_id mismatch"
    assert check["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert check["validation"]["metrics"]["ic"] == doc["validation"]["metrics"]["ic"]
    assert check["validation"]["metrics"]["icir"] == doc["validation"]["metrics"]["icir"]
    assert check["validation"]["metrics"]["max_abs_library_correlation"] == doc["validation"]["metrics"]["max_abs_library_correlation"]
    assert "data" in check["validation"]["signal_artifact"], "missing signal data"
    art = check["validation"]["signal_artifact"]
    print(f"VERIFIED {FID}: json ok, status={check['validation']['status']}, "
          f"IC={check['validation']['metrics']['ic']}, "
          f"ICIR={check['validation']['metrics']['icir']}, "
          f"artifact shape={art['shape']}")


if __name__ == "__main__":
    main()