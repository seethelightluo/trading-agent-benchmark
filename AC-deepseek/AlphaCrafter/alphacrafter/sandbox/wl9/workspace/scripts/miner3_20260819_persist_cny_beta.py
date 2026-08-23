"""miner_3 persistence (2026-08-19): persist cny_beta_60 (USDCNY-beta).

Rolling 60d beta of each tradable asset's daily return on USDCNY daily return
(renminbi-stress sensitivity). Rolling beta computed per asset on its own
calendar via concat-aligned asset/macro returns, then reindexed to the common
weekday grid for cross-sectional rank IC at h=10.

Admission (h=10, 15-asset universe): IC 0.0449, ICIR 0.1356 >= gates
0.0070/0.0840. max_abs_library_correlation 0.1452 (clean, no conflict with the
effective library). Persist with full-grid signal artifact (library convention).
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

LAST_VALIDATED = "2026-08-19"


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


def rolling_beta(asset_r, mkt_r, w, minp=40):
    df = pd.concat([asset_r.rename("a"), mkt_r.rename("m")], axis=1)
    beta = df["a"].rolling(w, min_periods=minp).cov(df["m"]) / df["m"].rolling(w, min_periods=minp).var()
    return beta


def main():
    closes = load_closes()
    macro = load_macro()
    ohlc = load_full_ohlc()
    print("assets loaded:", len(closes), "macro:", len(macro))

    fid = "cny_beta_60"
    usdcny = macro["USDCNY"].pct_change()
    vals = {a: rolling_beta(closes[a].pct_change(), usdcny, 60) for a in closes}
    res = evaluate(closes, vals, fid, horizon=10)
    print("max_abs_library_correlation:", res["max_abs_library_correlation"])

    if not res["passed"]:
        print("SKIP: gate fail")
        return
    if res["max_abs_library_correlation"] >= 0.5:
        print("SKIP: lib corr >= 0.5")
        return

    frame = full_grid_frame(ohlc, vals)
    artifact = make_artifact(frame)

    doc = {
        "factor_id": fid,
        "factor_name": "USDCNY Beta (60d)",
        "version": "1.0.0",
        "calculation": {
            "expression": "cov(r_asset, dUSDCNY, 60) / var(dUSDCNY, 60); dUSDCNY = pct_change(USDCNY close)",
            "description": (
                "Per-asset rolling 60d beta of daily return on USDCNY daily return "
                "(renminbi-stress sensitivity / China-currency-linkage). Higher positive "
                "cny-beta = asset amplifies CNH/CNY weakness; predictive of positive forward "
                "10-day returns at the cross-asset level."
            ),
        },
        "dependencies": ["close", "index_data"],
        "parameters": {"window": 60, "horizon": 10, "min_assets_for_ic": 8, "macro_series": "USDCNY"},
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": LAST_VALIDATED,
            "admission_horizon": 10,
            "regime_notes": (
                "Validated 2020-01-01..2026-07-15 (cross-asset universe, IC dates with >=8 "
                "valid assets). Covers COVID crash, 2020-21 bull, 2022 bear, 2023-24 AI rally, "
                "2024-26 crypto/commodity cycles. Cross-asset currency-linkage dimension "
                "(renminbi-stress beta), distinct from the DXY-correlation and VIX-beta factors "
                "in the library."
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
        "tags": ["macro", "currency", "beta", "linkage", "cross-asset"],
    }

    path = FACTOR_DIR / f"{fid}.json"
    path.write_text(json.dumps(doc))
    print(f"WROTE {path}")

    check = json.loads(path.read_text())
    assert check["factor_id"] == fid, "factor_id mismatch"
    assert check["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert check["validation"]["metrics"]["ic"] == doc["validation"]["metrics"]["ic"], "ic mismatch"
    assert check["validation"]["metrics"]["icir"] == doc["validation"]["metrics"]["icir"], "icir mismatch"
    assert check["validation"]["metrics"]["max_abs_library_correlation"] == doc["validation"]["metrics"]["max_abs_library_correlation"], "libcorr mismatch"
    assert "data" in check["validation"]["signal_artifact"], "artifact missing"
    print(
        f"VERIFIED {fid}: json ok, status={check['validation']['status']}, "
        f"IC={check['validation']['metrics']['ic']}, ICIR={check['validation']['metrics']['icir']}, "
        f"artifact shape={check['validation']['signal_artifact']['shape']}"
    )
    print("DONE")


if __name__ == "__main__":
    main()