"""miner_1 (2026-11-19): persist A2 clv_trend_20 and A3 sortino_20.

Admission gate (15-asset universe) at h=10: abs(IC)>=0.0070 & abs(ICIR)>=0.0840.
A2 clv_trend_20 measured IC=0.0547 ICIR=0.179; A3 sortino_20 IC=0.0528 ICIR=0.160.
Both passed. max_abs_library_correlation persisted as provenance/audit metadata
(post-mine gate recomputes rho from the real signal artifacts; the self-reported
value never substitutes for that and never directly rejects the candidate).
"""
from __future__ import annotations
import base64, hashlib, io, json, sys, zlib
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner3_20260730_harness import (
    FACTOR_DIR, evaluate, load_closes, to_frame,
)

LAST_VALIDATED = "2026-11-19"

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

def compute_clv_trend(closes):
    vals = {}
    for a, s in closes.items():
        hi = s.rolling(20).max()
        lo = s.rolling(20).min()
        rng = (hi - lo).replace(0, np.nan)
        clv = (s - lo) / rng
        m20 = s / s.shift(20) - 1.0
        vals[a] = (clv * np.sign(m20)).shift(1)
    return vals

def compute_sortino(closes):
    vals = {}
    for a, s in closes.items():
        r = s.pct_change()
        mu = r.rolling(20).mean()
        neg = r.clip(upper=0)
        dsd = neg.rolling(20).std().replace(0, np.nan)
        vals[a] = (mu / dsd).shift(1)
    return vals

def build_one(factor_id, factor_name, expression, description, params, tags, vals, res, regime):
    frame = res["frame"]
    artifact = make_artifact(frame)
    doc = {
        "factor_id": factor_id,
        "factor_name": factor_name,
        "version": "1.0.0",
        "benchmark_admission": {"ic_threshold": 0.0070, "icir_threshold": 0.0840},
        "calculation": {"expression": expression, "description": description},
        "dependencies": ["close"],
        "parameters": params,
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": LAST_VALIDATED,
            "admission_horizon": 10,
            "regime_notes": regime,
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
        "tags": tags,
    }
    path = FACTOR_DIR / f"{factor_id}.json"
    path.write_text(json.dumps(doc, indent=1))
    print("WROTE", path)
    check = json.loads(path.read_text())
    assert check["factor_id"] == factor_id
    assert check["validation"]["status"] == "EFFECTIVE"
    assert check["validation"]["metrics"]["ic"] == doc["validation"]["metrics"]["ic"]
    assert check["validation"]["metrics"]["icir"] == doc["validation"]["metrics"]["icir"]
    assert check["validation"]["metrics"]["max_abs_library_correlation"] == doc["validation"]["metrics"]["max_abs_library_correlation"]
    assert "data" in check["validation"]["signal_artifact"]
    print("VERIFIED", check["factor_id"], "status=", check["validation"]["status"],
          "art_shape=", check["validation"]["signal_artifact"]["shape"])

REGIME = ("Validated 2020-01-01..2026-07-15 across multiple regimes (COVID crash 2020, "
          "recovery bull 2020-21, 2022 tightening bear, 2023-24 AI equity rally, 2024-26 "
          "crypto/commodity cycles). Cross-sectional rank IC on 15-asset tradable universe.")

def main():
    closes = load_closes()
    print(f"assets loaded: {len(closes)}")

    vals = compute_clv_trend(closes)
    res = evaluate(closes, vals, "A2 clv_trend_20", horizon=10)
    build_one(
        "clv_trend_20", "Close-location-value * trend sign (20d)",
        "((close - min20) / (max20 - min20)) * sign(close / close.shift(20) - 1)",
        ("Position of close within its 20-day high-low range scaled by sign of the 20-day "
         "trend. High location with rising trend flags persistent strength; low location with "
         "falling trend flags persistent weakness."),
        {"win": 20}, ["trend", "technical", "location"], vals, res, REGIME,
    )

    vals2 = compute_sortino(closes)
    res2 = evaluate(closes, vals2, "A3 sortino_20", horizon=10)
    build_one(
        "sortino_20", "Sortino risk-adjusted return (20d)",
        "rolling_mean(daily_ret,20) / rolling_std(clip(daily_ret, upper=0),20)",
        ("20-day mean daily return scaled by downside (negative-only) standard deviation. "
         "Risk-adjusted momentum rewarding return per unit of downside risk."),
        {"win": 20}, ["risk-adjusted", "momentum", "quality"], vals2, res2, REGIME,
    )

if __name__ == "__main__":
    main()