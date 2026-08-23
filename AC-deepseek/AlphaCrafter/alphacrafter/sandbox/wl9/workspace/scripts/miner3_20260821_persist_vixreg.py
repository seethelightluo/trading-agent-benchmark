"""miner_3 persistence script (2026-08-21): persist mom_10_vixreg.

Sweep S confirmed vixreg_w10_s5_lb5 (5d momentum signed by sign of 10d VIX change
shifted 5d) is the most robust VIX-regime momentum variant:
  IC 0.0308, ICIR 0.0899 at h=10, max_abs_library_correlation 0.1021.
Split-half: first-half ICIR 0.0549, second-half ICIR 0.1258 (improving).

Writes factors/mom_10_vixreg.json with full-grid signal artifact, then verifies.
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
    load_macro,
)

LAST_VALIDATED = "2026-08-21"


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
    macro = load_macro()
    ohlc = load_full_ohlc()
    print("assets:", len(closes), "macro:", len(macro), "full ohlc:", len(ohlc))

    v = macro["VIX"].reindex(closes["SPX"].index)
    vr = v.pct_change(10)
    signv = pd.Series(
        np.where(vr.shift(5).notna(), np.where(vr.shift(5) > 0, -1.0, 1.0), np.nan),
        index=vr.index,
    )
    vals = {}
    for a in closes:
        mom = closes[a] / closes[a].shift(5) - 1.0
        vals[a] = mom * signv

    res = evaluate(closes, vals, "mom_10_vixreg", horizon=10)
    if not res["passed"]:
        print("SKIP: gate fail")
        return
    if res["max_abs_library_correlation"] >= 0.5:
        print("SKIP: lib corr >= 0.5")
        return

    frame = full_grid_frame(ohlc, vals)
    artifact = make_artifact(frame)

    doc = {
        "factor_id": "mom_10_vixreg",
        "factor_name": "VIX-Regime Filtered Short Momentum",
        "version": "1.0.0",
        "calculation": {
            "expression": "(close/close.shift(5)-1) * sign(VIX.change(10).shift(5))",
            "description": (
                "5-day momentum scaled by the sign of the VIX's 10-day change shifted 5 days ago "
                "(flipped: -1 when VIX rising, +1 when VIX falling). In risk-off regimes (VIX rising) "
                "short-horizon momentum is reversed/negated, while in risk-on regimes it is kept. "
                "Captures regime-conditional momentum across the 15-asset cross-section."
            ),
        },
        "dependencies": ["close", "macro:VIX"],
        "parameters": {
            "mom_lookback": 5,
            "vix_change_window": 10,
            "vix_shift_days": 5,
        },
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": LAST_VALIDATED,
            "admission_horizon": 10,
            "regime_notes": (
                "Validated 2020-01-01..2026-07-15 across COVID crash 2020, recovery bull, 2022 tightening "
                "bear, 2023-26 AI-equity rally and crypto/commodity cycles on the 15-asset tradable universe. "
                "Split-half: first-half ICIR 0.055, second-half ICIR 0.126 (improving). Tightly localized to "
                "the w10_s5_lb5 config; neighboring windows are submarginal."
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
        "tags": ["momentum", "regime", "cross-asset", "volatility-adaptive"],
    }

    path = FACTOR_DIR / "mom_10_vixreg.json"
    path.write_text(json.dumps(doc))
    print(f"WROTE {path}")

    # Verify
    check = json.loads(path.read_text())
    assert check["factor_id"] == "mom_10_vixreg", "factor_id mismatch"
    assert check["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert check["validation"]["metrics"]["ic"] == doc["validation"]["metrics"]["ic"]
    assert check["validation"]["metrics"]["icir"] == doc["validation"]["metrics"]["icir"]
    assert check["validation"]["metrics"]["max_abs_library_correlation"] == doc["validation"]["metrics"]["max_abs_library_correlation"]
    assert "data" in check["validation"]["signal_artifact"], "artifact missing"
    print(f"VERIFIED mom_10_vixreg: status={check['validation']['status']}, "
          f"IC={check['validation']['metrics']['ic']}, ICIR={check['validation']['metrics']['icir']}, "
          f"artifact shape={check['validation']['signal_artifact']['shape']}")


if __name__ == "__main__":
    main()