"""miner_2 persistence (2026-08-19): persist passing candidates from sweep I.

Passing candidates (abs(IC)>=0.0070 & abs(ICIR)>=0.0840 at h=10 on the 15-asset
universe) with max_abs_library_correlation < 0.5:
  - kaufman_eff_20d  (IC 0.0497, ICIR 0.1655, max_corr 0.3163) -- trend efficiency
  - kurt_20d         (IC 0.0267, ICIR 0.0856, max_corr 0.1695) -- return kurtosis
  (vol_z_20d_v2 is a duplicate of persisted vol_z_20d; skipped by design.)

Builds full-grid signal artifacts (library convention), writes
factors/<fid>.json, verifies by reading back, and also computes a
routine-revalidation report of all currently effective factors.
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

LAST_VALIDATED = "2026-08-19"


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


def kaufman_eff(close, n=20):
    """Kaufman efficiency ratio: |close-close_n| / sum(|diffs|) in n days."""
    num = (close - close.shift(n)).abs()
    den = close.diff().abs().rolling(n).sum().replace(0, np.nan)
    return num / den


def kurt_20d(close, n=20, minp=10):
    """Excess kurtosis of daily returns over n days."""
    r = close.pct_change()
    m = r.rolling(n, min_periods=minp).mean()
    sd = r.rolling(n, min_periods=minp).std(ddof=0).replace(0, np.nan)
    m4 = ((r - m) ** 4).rolling(n, min_periods=minp).mean()
    return m4 / (sd ** 4) - 3.0


def persist_one(closes, ohlc, fid, spec, direction):
    vals = {a: spec["vals_fn"](a) for a in closes}
    res = evaluate(closes, vals, fid, horizon=10)
    print(f"  evaluated {fid}: IC={res['ic']:.4f} ICIR={res['icir']:.4f} "
          f"max_corr={res['max_abs_library_correlation']:.4f} passed={res['passed']}")

    if not res["passed"]:
        print(f"  SKIP {fid}: gate fail")
        return False
    if res["max_abs_library_correlation"] >= 0.5:
        print(f"  SKIP {fid}: lib corr {res['max_abs_library_correlation']:.3f} >= 0.5")
        return False

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
        "expected_direction": direction,
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
    print(f"  WROTE {path}")

    check = json.loads(path.read_text())
    assert check["factor_id"] == fid, "factor_id mismatch"
    assert check["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert check["validation"]["metrics"]["ic"] == doc["validation"]["metrics"]["ic"]
    assert check["validation"]["metrics"]["icir"] == doc["validation"]["metrics"]["icir"]
    assert check["validation"]["metrics"]["max_abs_library_correlation"] == doc["validation"]["metrics"]["max_abs_library_correlation"]
    assert "data" in check["validation"]["signal_artifact"], "artifact missing"
    shape = check["validation"]["signal_artifact"]["shape"]
    print(f"  VERIFIED {fid}: json ok, status={check['validation']['status']}, "
          f"IC={check['validation']['metrics']['ic']}, ICIR={check['validation']['metrics']['icir']}, "
          f"artifact shape={shape}")
    return True


def revalidate_all(closes):
    """Routine re-validation of currently effective factors; report drift."""
    print("\n=== ROUTINE RE-VALIDATION OF EFFECTIVE FACTORS ===")
    for f in sorted(FACTOR_DIR.glob("*.json")):
        if f.name == "factor_ensemble.json":
            continue
        try:
            d = json.load(open(f))
        except Exception:
            continue
        if d.get("validation", {}).get("status") != "EFFECTIVE":
            continue
        fid = d["factor_id"]
        art = d["validation"].get("signal_artifact", {})
        if not art or "data" not in art:
            continu