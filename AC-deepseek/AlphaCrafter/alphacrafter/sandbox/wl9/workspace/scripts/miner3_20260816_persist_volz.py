"""miner_3 persistence (2026-08-16): persist vol_z_20d.

Volume-participation z-score: (volume - SMA20(volume)) / STD20(volume) per asset,
computed cross-sectionally across the 9 tradable assets with real volume data
(equity indices + crypto); other 6 assets (SOX/XAU/COPPER/WTI/US10Y/CN10Y) have
constant volume -> NaN.

Admission (h=10, 15-asset universe): IC 0.0505, ICIR 0.1200 >= gates 0.0070/0.0840.
Robustness: no-crypto IC 0.0461 / ICIR 0.0975; 2020-23 IC 0.0777 / ICIR 0.1845,
2023-26 IC 0.0195 / ICIR 0.0467; max_abs_library_correlation 0.1138.
Persist with full-grid signal artifact (library convention).
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
VOL_ASSETS = ["000300.SH", "000688.SH", "SPX", "HSI", "N225", "SX5E", "NDX", "BTC", "ETH"]


def load_full_ohlc():
    out = {}
    for a in ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        if not f.exists():
            continue
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        out[a] = df.set_index("date")
    return out


def load_full_volumes():
    out = {}
    for a in VOL_ASSETS:
        f = STOCK_DIR / f"{a}.csv"
        df = pd.read_csv(f, parse_dates=["date"]).sort_values("date")
        out[a] = df.set_index("date")["volume"].astype(float)
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


def vol_z(volume, n=20):
    mu = volume.rolling(n).mean()
    sd = volume.rolling(n).std(ddof=0).replace(0, np.nan)
    return (volume - mu) / sd


def main():
    closes = load_closes()
    ohlc = load_full_ohlc()
    vols = load_full_volumes()
    print("assets loaded:", len(closes), "volume series:", len(vols))

    fid = "vol_z_20d"
    vals = {
        a: vol_z(vols[a], 20)
        if a in vols
        else pd.Series(np.nan, index=closes[a].index)
        for a in closes
    }
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
        "factor_name": "Volume Participation Z-Score 20d",
        "version": "1.0.0",
        "calculation": {
            "expression": "(volume - rolling_mean(volume,20)) / rolling_std(volume,20)",
            "description": (
                "Per-asset z-score of current volume vs its trailing 20-day mean/std. High positive "
                "z = unusual volume expansion (participation surge); predictive of positive forward "
                "10-day returns at the cross-asset level. Computed on the 9 tradable assets with real "
                "volume data; constant-volume assets (SOX, XAU, COPPER, WTI, US10Y, CN10Y) are NaN."
            ),
        },
        "dependencies": ["volume"],
        "parameters": {"window": 20, "volume_assets": VOL_ASSETS},
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": LAST_VALIDATED,
            "admission_horizon": 10,
            "regime_notes": (
                "Validated 2020-01-01..2026-07-15 (1706 factor dates, 1493 IC dates with >=8 valid "
                "assets). Covers COVID crash, 2020-21 bull, 2022 bear, 2023-24 AI rally, 2024-26 "
                "crypto/commodity cycles. Sub-period 2020-01..2023-06: IC 0.0777/ICIR 0.1845; "
                "2023-07..2026-07: IC 0.0195/ICIR 0.0467 (positive but weaker). Excluding BTC/ETH: "
                "IC 0.0461/ICIR 0.0975, confirming signal is not crypto-driven. Rolling 1y ICs "
                "positive in 5 of 6 yearly windows (only dip: 2023H2, IC -0.0395)."
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
                "excl_crypto_ic": 0.0461,
                "excl_crypto_icir": 0.0975,
            },
            "signal_artifact": artifact,
        },
        "tags": ["liquidity", "volume", "participation", "cross-asset"],
    }

    path = FACTOR_DIR / f"{fid}.json"
    path.write_text(json.dumps(doc))
    print(f"WROTE {path}")

    check = json.loads(path.read_text())
    assert check["factor_id"] == fid, "factor_id mismatch"
    assert check["validation"]["status"] == "EFFECTIVE", "status mismatch"
    assert check["validation"]["metrics"]["ic"] == doc["validation"]["metrics"]["ic"]
    assert check["validation"]["metrics"]["icir"] == doc["validation"]["metrics"]["icir"]
    assert check["validation"]["metrics"]["max_abs_library_correlation"] == doc["validation"]["metrics"]["max_abs_library_correlation"]
    assert "data" in check["validation"]["signal_artifact"], "artifact missing"
    print(
        f"VERIFIED {fid}: json ok, status={check['validation']['status']}, "
        f"IC={check['validation']['metrics']['ic']}, ICIR={check['validation']['metrics']['icir']}, "
        f"artifact shape={check['validation']['signal_artifact']['shape']}"
    )
    print("DONE")


if __name__ == "__main__":
    main()