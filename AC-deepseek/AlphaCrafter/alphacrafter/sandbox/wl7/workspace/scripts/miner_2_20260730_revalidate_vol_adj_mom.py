"""miner_2 2026-07-30: re-validate vol_adj_mom_20x60 (passed cycle4 but was NOT persisted)
and persist it with a recoverable signal artifact (base64:zlib:csv) to close the gap."""
from __future__ import annotations
import sys, json, base64, zlib, io, csv, hashlib
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_2_lib import (validate_factor, load_panel, load_macro, per_asset,
                         FACTOR_LAST, ADMISSION)


@per_asset
def vol_adj_mom_20x60(s: pd.Series) -> pd.Series:
    """20d momentum (skip 5d) scaled by 60d realized vol: trend strength per unit risk."""
    mom = s.shift(5) / s.shift(25) - 1.0
    vol = s.pct_change().rolling(60).std()
    return mom / vol


def build_artifact(panel: pd.DataFrame, factor: pd.DataFrame) -> dict:
    """CSV artifact: rows=dates, cols=assets, truncated to validated warm-up window."""
    f = factor.loc[:FACTOR_LAST].reindex(panel.index)
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["date"] + list(f.columns))
    for dt in f.index:
        w.writerow([dt.date().isoformat()] + ["" if pd.isna(v) else f"{float(v):.12g}" for v in f.loc[dt]])
    raw = buf.getvalue().encode()
    comp = zlib.compress(raw, level=6)
    b64 = base64.b64encode(comp).decode()
    n_valid = int(f.notna().sum().sum())
    return {
        "format": "base64:zlib:csv",
        "description": f"Factor signal panel: rows = dates, cols = assets. Shape {list(f.shape)}",
        "columns": list(f.columns),
        "shape": list(f.shape),
        "n_valid_values": n_valid,
        "sha256": hashlib.sha256(comp).hexdigest()[:16],
        "data": b64,
    }


def main():
    panel = load_panel()
    macro = load_macro()
    factor = vol_adj_mom_20x60(panel, macro)
    r = validate_factor("vol_adj_mom_20x60", vol_adj_mom_20x60, direction_override=1.0)
    print(json.dumps({"admission_gate": r["admission_gate"], "ic_h10": r["ic_h10"],
                      "icir_h10": r["icir_h10"], "hit_h10": r["hit_h10"],
                      "n_dates_h10": r["n_dates_h10"], "max_corr": r["max_abs_library_correlation"],
                      "library_corrs": r["library_corrs"]}, indent=1))

    if not r["admission_gate"]["pass"]:
        print("NOT PASSING -> skip persistence")
        return

    factor_id = "vol_adj_mom_20x60"
    payload = {
        "factor_id": factor_id,
        "factor_name": "Volatility-adjusted momentum 20d/60d (risk-scaled trend)",
        "version": "1.0.0",
        "calculation": {
            "expression": "(close.shift(5)/close.shift(25)-1) / rolling_std(pct_change(close),60)",
            "description": "20-day momentum (5-day skip) divided by 60-day realized volatility: "
                           "trend strength per unit of risk. Favours assets with strong, steady trends "
                           "rather than raw return; a vol-scaled momentum premium.",
        },
        "dependencies": ["close"],
        "parameters": {"mom_lookback": 20, "skip": 5, "vol_window": 60, "min_periods": 30},
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-15",
            "last_validated": "2026-07-30",
            "admission_horizon": 10,
            "regime_notes": "Re-validated 2020-01-01..2026-07-15 (15 cross-asset instruments). "
                            "h10 IC/ICIR pass admission; decay peaks at h10-20. Corr with library: " +
                            json.dumps(r["library_corrs"]),
            "metrics": {
                "ic": r["ic_h10"],
                "icir": r["icir_h10"],
                "ic_hit_ratio": r["hit_h10"],
                "n_ic_dates": r["n_dates_h10"],
                "coverage_asset_days": r["coverage_asset_days"],
                "coverage_dates_ge8": r["coverage_dates_ge8"],
                "turnover_10d_rank": r["turnover_10d_rank"],
                "decay_ic_by_horizon": r["decay_ic_by_horizon"],
                "max_abs_library_correlation": r["max_abs_library_correlation"],
            },
            "signal_artifact": build_artifact(panel, factor),
        },
        "tags": ["momentum", "volatility-adjusted", "trend", "risk-scaled"],
        "benchmark_admission": {
            "contract": {"ic_threshold": ADMISSION["ic"], "icir_threshold": ADMISSION["icir"],
                         "correlation_threshold": 0.5},
            "selected_metrics": {
                "ic": r["ic_h10"], "icir": r["icir_h10"],
                "metric_path": "validation.metrics",
                "max_abs_library_correlation": r["max_abs_library_correlation"],
                "correlation_path": "validation.metrics.max_abs_library_correlation",
            },
        },
    }
    path = Path("factors") / f"{factor_id}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(f"PERSISTED -> {path} ({path.stat().st_size} bytes)")

    # round-trip verification
    back = json.loads(path.read_text())
    a = back["validation"]["signal_artifact"]
    dec = zlib.decompress(base64.b64decode(a["data"])).decode()
    rows = list(csv.reader(io.StringIO(dec)))
    print("VERIFY: id=", back["factor_id"], "status=", back["validation"]["status"],
          "rows=", len(rows) - 1, "cols=", len(rows[0]),
          "sha match=", hashlib.sha256(base64.b64decode(a["data"])).hexdigest()[:16] == a["sha256"])
    json.dump({"vol_adj_mom_20x60": {"ic_h10": r["ic_h10"], "icir_h10": r["icir_h10"],
                                     "pass": True, "persisted": True}},
              open("scripts/miner_2_cycle5_results.json", "w"), indent=1)


if __name__ == "__main__":
    main()
