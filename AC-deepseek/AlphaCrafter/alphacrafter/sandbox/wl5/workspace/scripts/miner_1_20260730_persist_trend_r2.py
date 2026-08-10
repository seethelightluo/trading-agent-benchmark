"""Persist validated factor trend_r2_30_signed (miner_1, 2026-07-30).

Admission metrics come from scripts/miner_1_20260730_explore_trend_r2_results.json
(executed 2026-07-30, data through 2026-07-29):
  ic=0.0562, icir=0.1672 (gates: |ic|>=0.007, |icir|>=0.084).
"""
import json
import os
from datetime import datetime, timezone

FACTOR_ID = "trend_r2_30_signed"

results = json.load(open("scripts/miner_1_20260730_explore_trend_r2_results.json"))
m = results[FACTOR_ID]

factor = {
    "factor_id": FACTOR_ID,
    "factor_name": "Signed 30-day Trend R2 (log-price fit quality)",
    "version": "1.0.0",
    "calculation": {
        "expression": (
            "sign(cov) * cov^2 / (var(t) * var(log_close)) over trailing 30d, "
            "where cov = rolling cov(log_close, t); result is the signed R^2 of "
            "an OLS fit of log-price on time, sign = direction of the slope."
        ),
        "description": (
            "Measures how strongly and in which direction an asset's log-price "
            "follows a straight-line trend over the past 30 trading days. "
            "Positive values indicate clean up-trends, negative values clean "
            "down-trends; near-zero means choppy/range-bound price action. "
            "Computed vectorized from log-close on the 15-asset tradable "
            "cross-asset universe; minimum 18 of 30 observations required."
        ),
    },
    "dependencies": ["close"],
    "parameters": {
        "window": 30,
        "min_periods_ratio": 0.6,
        "min_periods": 18,
        "admission_horizon": 10,
    },
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-29",
        "last_validated": "2026-07-30",
        "admission_horizon": 10,
        "regime_notes": (
            "Cross-sectional rank IC vs 10-day forward returns on the 15-asset "
            "tradable universe (equity indices, commodities, crypto, yield "
            "series). Regime split: 2020-2022 IC=0.0803 (ICIR=0.246, n=442) "
            "strong during COVID crash/recovery and 2022 tightening bear; "
            "2023-2024 IC=0.0047 (ICIR=0.014, n=305) flat during the "
            "AI-led equity rally / range-bound regime; 2025-2026 IC=0.0777 "
            "(ICIR=0.2175, n=234) strong again. Non-monotonic across regimes, "
            "but overall stable with 10d IC hit ratio 0.562 and decay IC rising "
            "from 0.0246 (1d) to 0.0646 (20d), i.e. signal strengthens with "
            "holding horizon. Re-validate every ~3 months."
        ),
        "metrics": {
            "ic": m["ic"],
            "icir": m["icir"],
            "ic_hit_ratio": m["ic_hit_ratio"],
            "n_ic_dates": m["n_ic_dates"],
            "coverage_asset_days": m["coverage_asset_days"],
            "coverage_dates_ge8": m["coverage_dates_ge8"],
            "turnover_10d_rank": m["turnover_10d_rank"],
            "decay_ic_by_horizon": m["decay_ic_by_horizon"],
            "max_abs_library_correlation": m["max_abs_library_correlation"],
        },
    },
    "tags": ["trend", "quality", "cross-asset", "time-series"],
    "benchmark_admission": {
        "contract": {
            "ic_threshold": 0.007,
            "icir_threshold": 0.084,
            "correlation_threshold": 0.5,
            "library_capacity": 30,
            "active_top_k": 10,
        },
        "selected_metrics": {
            "ic": m["ic"],
            "icir": m["icir"],
            "metric_path": "validation.metrics",
            "max_abs_library_correlation": m["max_abs_library_correlation"],
            "correlation_path": "validation.metrics.max_abs_library_correlation",
        },
        "admitted_at": datetime.now(timezone.utc).isoformat(),
    },
}

out_path = f"factors/{FACTOR_ID}.json"
with open(out_path, "w") as f:
    json.dump(factor, f, indent=1)

# Read back and verify
with open(out_path) as f:
    back = json.load(f)
assert back["factor_id"] == FACTOR_ID, "factor_id mismatch"
assert back["validation"]["status"] == "EFFECTIVE", "status not EFFECTIVE"
assert abs(back["validation"]["metrics"]["ic"]) >= 0.007, "ic below gate"
assert abs(back["validation"]["metrics"]["icir"]) >= 0.084, "icir below gate"
assert "max_abs_library_correlation" in back["validation"]["metrics"]
assert "decay_ic_by_horizon" in back["validation"]["metrics"]
print("PERSISTED", out_path)
print("  ic      =", back["validation"]["metrics"]["ic"])
print("  icir    =", back["validation"]["metrics"]["icir"])
print("  status  =", back["validation"]["status"])
print("  gates   =", back["benchmark_admission"]["contract"])
print("  max_rho =", back["validation"]["metrics"]["max_abs_library_correlation"])
print("  decay   =", back["validation"]["metrics"]["decay_ic_by_horizon"])
print("VERIFY OK: valid JSON, id/status/thresholds/metrics present and reloadable.")
