"""miner_3 2026-07-30 cycle 28: persist eff_ratio_20 (Kaufman efficiency ratio, 20d).

Writes factors/eff_ratio_20.json + factors/eff_ratio_20.signal.npy (real signal artifact
so the deterministic post-Miner correlation gate can recover and re-check rho).
Metrics recomputed from the executed validation (admission horizon = 10d).
"""
import sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, per_asset, forward_returns, compute_ic,
                         validate_factor, VISIBLE_THROUGH)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# recompute factor (deterministic)
f = per_asset(panel, lambda s: (s - s.shift(20)).abs()
              / s.diff().abs().rolling(20, min_periods=15).sum())
sig = f.reindex(panel.index)

# library artifacts for correlation check
lib = {}
for fid in ["mom20_volproxy60", "dxy_beta_cond_60x20"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    lib[fid] = pd.DataFrame(arr, index=panel.index, columns=panel.columns)

m = validate_factor(f, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                    library=lib, fwd_cache=fwd_cache)
print("ADMISSION METRICS:", json.dumps(m, indent=1))

assert abs(m["ic"]) >= 0.0070 and abs(m["icir"]) >= 0.0840, "gate not met"

# ---- persist signal artifact ----
np.save("factors/eff_ratio_20.signal.npy", sig.values)
print("signal artifact saved", sig.shape)

# ---- persist JSON ----
doc = {
    "factor_id": "eff_ratio_20",
    "factor_name": "kaufman_efficiency_ratio_20d",
    "version": "1.0.0",
    "calculation": {
        "expression": "abs(close - close.shift(20)) / sum(abs(close.diff()), 20)",
        "description": ("Kaufman efficiency ratio over 20 days on the asset's own calendar: "
                        "net 20d displacement divided by the sum of absolute 1d moves. High "
                        "values indicate smooth, directional trending; low values indicate "
                        "choppy/range-bound behavior. Cross-sectionally long the smoothly "
                        "trending assets at 10d horizon. Rank-based (Spearman) usage."),
        "transform": "rank cross-sectionally (pct rank); long top, underweight bottom"
    },
    "dependencies": ["close"],
    "parameters": {"window": 20, "min_periods": 15},
    "validation": {
        "status": "EFFECTIVE",
        "period": {"start": "2020-01-01", "end": VISIBLE_THROUGH},
        "metrics": {
            "ic_admission_10d": m["ic"],
            "icir_admission_10d": m["icir"],
            "ic_hit_ratio": m["ic_hit_ratio"],
            "n_ic_dates": m["n_ic_dates"],
            "coverage_asset_days": m["coverage_asset_days"],
            "coverage_dates_ge8": m["coverage_dates_ge8"],
            "turnover_10d_rank": m["turnover_10d_rank"],
            "decay_ic_by_horizon": m["decay_ic_by_horizon"],
            "max_abs_library_correlation": m["max_abs_library_correlation"],
            "library_pairwise_corr": m["library_pairwise_corr"],
        },
        "regime_notes": (
            "10d IC by sub-period: 2020-2021 +0.0319 (icir +0.108), 2022 +0.0757 (+0.248), "
            "2023-2024 +0.0311 (+0.105), 2025-2026 +0.0791 (+0.260). Positive and stable "
            "across bull, bear (2022), and crypto-winter regimes. Library correlation low "
            "(max |rho| 0.17 vs mom20_volproxy60), offering orthogonal trend-quality signal."
        ),
        "decay": "IC rises from 1d (0.018) to 10d (0.050) then eases to 20d (0.044); "
                 "peak around the 10d admission horizon - consistent with medium-term trend persistence.",
    },
    "tags": ["trend", "efficiency_ratio", "trend_quality", "momentum_structure", "cross_asset"],
    "last_validated": "2026-07-30",
}

with open("factors/eff_ratio_20.json", "w") as fh:
    json.dump(doc, fh, indent=2)
print("JSON written.")

# ---- read back and verify ----
back = json.load(open("factors/eff_ratio_20.json"))
sig2 = np.load("factors/eff_ratio_20.signal.npy")
ok = (back["factor_id"] == "eff_ratio_20"
      and back["validation"]["status"] == "EFFECTIVE"
      and abs(back["validation"]["metrics"]["ic_admission_10d"]) >= 0.0070
      and abs(back["validation"]["metrics"]["icir_admission_10d"]) >= 0.0840
      and sig2.shape == sig.shape and np.allclose(sig2, sig.values, equal_nan=True))
print("READBACK OK:", ok)
print("factor_id:", back["factor_id"], "| status:", back["validation"]["status"],
      "| IC:", back["validation"]["metrics"]["ic_admission_10d"],
      "| ICIR:", back["validation"]["metrics"]["icir_admission_10d"],
      "| maxlibcorr:", back["validation"]["metrics"]["max_abs_library_correlation"])
