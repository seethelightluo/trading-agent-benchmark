"""miner_3 focused validation (2026-08-19): cny_beta_60 (USDCNY-beta).

Rolling 60d beta of each tradable asset's daily return on USDCNY daily return
(renminbi-stress sensitivity). Rolling beta is computed per asset on its own
calendar, then reindexed to the common weekday grid for cross-sectional rank IC
at h=10.

Full evaluation + sub-period robustness + persistence with full-grid signal
artifact (library convention).
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
    load_macro,
)
import miner3_20260730_harness as H  # noqa: E402

LAST_VALIDATED = "2026-08-19"


def rolling_beta(asset_r: pd.Series, mkt_r: pd.Series, w: int, minp: int = 40) -> pd.Series:
    df = pd.concat([asset_r.rename("a"), mkt_r.rename("m")], axis=1)
    beta = (
        df["a"].rolling(w, min_periods=minp).cov(df["m"])
        / df["m"].rolling(w, min_periods=minp).var().replace(0, np.nan)
    )
    return beta


def full_grid_frame(closes, values, start="2020-01-01", end=None):
    """Full (non-capped) weekday grid artifact frame: rows=all weekdays with any
    data, cols=15 assets. End defaults to the latest visible data date."""
    dates = sorted({d for s in closes.values() for d in s.index})
    if end is not None:
        dates = [d for d in dates if d <= pd.Timestamp(end)]
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


def subperiod_stats(ic_series, cut="2023-07-01"):
    """Split IC series at cut and compute mean/ICIR for each half."""
    out = {}
    early = ic_series[ic_series.index < pd.Timestamp(cut)]
    late = ic_series[ic_series.index >= pd.Timestamp(cut)]
    for name, s in (("early_2020_2023H1", early), ("late_2023H2_2026", late)):
        if len(s) < 30:
            out[name] = {"ic": float("nan"), "icir": float("nan"), "n": int(len(s))}
            continue
        m = float(s.mean())
        sd = float(s.std(ddof=1))
        out[name] = {"ic": round(m, 4), "icir": round(m / sd, 4) if sd else float("nan"), "n": int(len(s))}
    return out


def main():
    closes = load_closes()
    macro = load_macro()
    print("assets loaded:", len(closes), "macro loaded:", len(macro))
    usdcny_r = macro["USDCNY"].pct_change()

    fid = "cny_beta_60"
    vals = {a: rolling_beta(closes[a].pct_change(), usdcny_r, 60) for a in closes}
    res = evaluate(closes, vals, fid, horizon=10, verbose=True)
    print()
    print("max_abs_library_correlation:", round(res["max_abs_library_correlation"], 4))
    print("sub-period:", subperiod_stats(res["ic_series"]))

    # year-window IC stability (calendar-year means)
    yr = res["ic_series"].groupby(res["ic_series"].index.year).mean()
    print("yearly mean IC:", {str(k): round(float(v), 4) for k, v in yr.items()})

    if not res["passed"]:
        print("SKIP: gate fail")
        return
    if res["max_abs_library_correlation"] >= 0.5:
        print("SKIP: lib corr >= 0.5")
        return

    # full-grid artifact through latest visible data (consistent with library files)
    full = full_grid_frame(closes, vals)
    artifact = make_artifact(full)

    doc = {
        "factor_id": fid,
        "factor_name": "USDCNY beta (60d)",
        "version": "1.0.0",
        "calculation": {
            "expression": "rolling_beta(r_asset, r_USDCNY, 60), r = pct_change(close)",
            "description": (
                "Rolling 60d beta of each tradable asset's daily return on USDCNY daily return. "
                "A high positive value means the asset tends to rise when the renminbi weakens "
                "(dollar strengthens vs CNY) - i.e. renminbi-stress / China-risk sensitivity. "
                "Validated POSITIVE 10d direction: assets with high CNY-stress beta tend to "
                "outperform over the next 10 trading days (renminbi-stress risk premium). "
                "Distinct from DXY-beta family: correlation with beta_VIX_60 is only -0.145 "
                "and with dxy_corr_change_20_60 only +0.027."
            ),
        },
        "dependencies": ["close", "index_data"],
        "parameters": {
            "window": 60,
            "min_periods": 40,
            "horizon": 10,
            "min_assets_for_ic": 8,
            "macro_series": "USDCNY",
        },
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": LAST_VALIDATED,
            "admission_horizon": 10,
            "regime_notes": (
                "Validated