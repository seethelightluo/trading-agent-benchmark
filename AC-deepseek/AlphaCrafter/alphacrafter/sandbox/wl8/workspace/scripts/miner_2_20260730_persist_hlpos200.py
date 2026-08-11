"""miner_2 2026-07-30 -- persist hl_pos_200 (best variant of the range-position family)."""
import sys
import json
import hashlib
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from factor_validation_lib import (load_closes, load_index, factor_panel,
                                   fwd_returns, ic_series, validate_factor,
                                   load_library_panels, max_library_corr,
                                   artifact_b64, IC_GATE, ICIR_GATE, ASSETS)

close, vol, open_, high, low = load_closes()
vix = load_index("VIX")
macro = {"VIX": vix, "DXY": load_index("DXY")}
macro["US10Y"] = close["US10Y"].dropna()
macro["CN10Y"] = close["CN10Y"].dropna()

lib = load_library_panels()
print(f"library panels loaded: {list(lib.keys())}", flush=True)


def hl_pos(c, v, o, h, l, m, w, skip=0):
    hi = h.rolling(w).max().shift(skip)
    lo = l.rolling(w).min().shift(skip)
    rng = (hi - lo).replace(0, np.nan)
    return (c.shift(skip) - lo) / rng


factor_id = "hl_pos_200"
panel = factor_panel(hl_pos, close, vol, open_, high, low, macro, w=200, skip=0)
res = validate_factor(hl_pos, close, vol, open_, high, low, macro, w=200, skip=0)
res["max_abs_library_correlation"] = max_library_corr(panel, lib)
ic10 = ic_series(panel, fwd_returns(close, 10))
ic = float(ic10.mean()); icir = float(ic10.mean() / ic10.std())
ok = abs(ic) >= IC_GATE and abs(icir) >= ICIR_GATE
print(f"=== {factor_id} ===", flush=True)
print(f"  ic={ic:.4f} icir={icir:.4f} hit={res['ic_hit_ratio']:.3f} "
      f"n={res['n_ic_dates']} cov={res['coverage_asset_days']:.3f}/{res['coverage_dates_ge8']:.2f} "
      f"to={res['turnover_10d_rank']:.2f}", flush=True)
print(f"  decay={res['decay_ic_by_horizon']}", flush=True)
print(f"  max_abs_library_correlation={res['max_abs_library_correlation']:.4f}", flush=True)
regs = {"2020 COVID": ("2020-01-01", "2020-12-31"), "2021 bull": ("2021-01-01", "2021-12-31"),
        "2022 tightening": ("2022-01-01", "2022-12-31"), "2023 recovery": ("2023-01-01", "2023-12-31"),
        "2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"),
        "2026H1": ("2026-01-01", "2026-06-30"), "2026 recent": ("2026-04-01", "2026-07-30")}
print("  regime IC (h=10):", flush=True)
for rname, (a, b) in regs.items():
    sub = ic10.loc[(ic10.index >= a) & (ic10.index <= b)]
    if len(sub):
        print(f"    {rname}: ic={sub.mean():.4f} icir={sub.mean()/sub.std():.3f} n={len(sub)}", flush=True)
print(f"  GATE: {'PASS' if ok else 'FAIL'}", flush=True)

if ok and res["max_abs_library_correlation"] < 0.5:
    d = {
        "factor_id": factor_id,
        "factor_name": "High-Low Range Position 200d",
        "version": "1.0.0",
        "calculation": {
            "expression": "(close - rolling_min(low,200)) / (rolling_max(high,200) - rolling_min(low,200))",
            "description": "Price location within the trailing 200-day high-low range: 1.0 at range top, 0.0 at range bottom. "
                           "Slow trend-following price-location signal; positive direction (near range top -> continue up).",
        },
        "dependencies": ["close", "high", "low"],
        "parameters": {"lookback": 200, "skip": 0},
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-30",
            "last_validated": "2026-07-30",
            "admission_horizon": 10,
            "regime_notes": "Validated 2020-01-01..2026-07-30 on the 15-asset tradable cross-asset universe. "
                            "Horizon-10 rank IC positive in 2021 bull, 2022 tightening, 2024, 2025, 2026H1 and 2026 "
                            "recent sub-periods (strong 0.20 recent); negative only in 2020 COVID window and flat 2023. "
                            "Slowest member of the hl_pos family; max|rho| vs effective library < 0.5.",
            "metrics": {
                "ic": res["ic"], "icir": res["icir"], "ic_hit_ratio": res["ic_hit_ratio"],
                "n_ic_dates": res["n_ic_dates"], "coverage_asset_days": res["coverage_asset_days"],
                "coverage_dates_ge8": res["coverage_dates_ge8"],
                "turnover_10d_rank": res["turnover_10d_rank"],
                "decay_ic_by_horizon": res["decay_ic_by_horizon"],
                "max_abs_library_correlation": res["max_abs_library_correlation"],
            },
            "signal_artifact": {
                "format": "base64:zlib:csv",
                "description": "Factor signal panel: rows = dates, cols = assets",
                "columns": list(panel.columns),
                "shape": [int(panel.shape[0]), int(panel.shape[1])],
                "n_valid_values": int(panel.notna().sum().sum()),
                "sha256": hashlib.sha256(panel.to_csv().encode()).hexdigest()[:16],
                "data": artifact_b64(panel),
            },
        },
        "tags": ["price-location", "range", "trend", "technical"],
    }
    with open(f"factors/{factor_id}.json", "w") as f:
        json.dump(d, f, indent=1)
    print(f"[persisted] factors/{factor_id}.json", flush=True)
print("done", flush=True)
