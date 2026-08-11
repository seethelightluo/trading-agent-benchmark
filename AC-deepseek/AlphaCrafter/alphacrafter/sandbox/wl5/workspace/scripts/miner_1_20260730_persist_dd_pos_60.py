"""miner_1 cycle 2026-07-30: persist dd_pos_60 (60d range-position / drawdown-cycle factor).

dd_pos_60 = (close - rolling_min(close,60)) / (rolling_max(close,60) - rolling_min(close,60))
Motivation: measures where an asset sits within its recent 60-day high-low range. High values
(at/near the range top) capture recovery strength / breakout continuation; low values capture
assets in the early stage of drawdowns. Complements existing library factors which use trend
quality (trend_r2_30_signed), downside volatility (semi_down_ratio_20), and time-under-water
(time_under_water_120) but not raw range position at the 60d horizon.

Validation: cross-sectional Spearman rank IC vs 10d forward returns, 15-asset tradable
universe, visible window <= 2026-07-29. Gate: |IC| >= 0.007, |ICIR| >= 0.084.
Signal artifact stored for deterministic post-Miner gate rho recomputation.
"""
import json, sys, time, os
import numpy as np
import pandas as pd
sys.path.insert(0, 'scripts')
from factor_validate import (closes_panel, forward_returns, ic_series,
                             summary_metrics, regime_split)
sys.path.insert(0, 'scripts')
from miner3_lib import library_max_rho, build_artifact

VIS = '2026-07-29'
H = 10
t0 = time.time()
close = closes_panel(VIS)
print(f"panel: dates={len(close)} assets={len(close.columns)} visible_through={VIS} load={time.time()-t0:.1f}s", flush=True)

# ---- signal ----
roll_max60 = close.rolling(60, min_periods=36).max()
roll_min60 = close.rolling(60, min_periods=36).min()
sig = (close - roll_min60) / (roll_max60 - roll_min60)

fr = forward_returns(close, H)
ic_s = ic_series(sig, fr, min_valid=8)
m = summary_metrics(ic_s, sig, fr, close, h=H)
m['regime'] = regime_split(ic_s)

# full-library rho against persisted signal artifacts
rhos, maxrho = library_max_rho(sig)
m['library_rho_by_factor'] = rhos
m['max_abs_library_correlation'] = round(maxrho, 3)
print("library rho by factor:", json.dumps(rhos, indent=1), flush=True)

gate_ic = abs(m['ic']) >= 0.007
gate_icir = abs(m['icir'] or 0) >= 0.084
gate = bool(gate_ic and gate_icir)
print(f"=== dd_pos_60: ic={m['ic']} icir={m['icir']} hit={m['ic_hit_ratio']} n={m['n_ic_dates']} "
      f"cov_ad={m['coverage_asset_days']} cov_ge8={m['coverage_dates_ge8']} "
      f"turn={m['turnover_10d_rank']} max_rho_lib={maxrho} GATE={gate}", flush=True)
print("  decay:", m['decay_ic_by_horizon'], flush=True)
print("  regimes:", m['regime'], flush=True)

if not gate:
    print("GATE FAILED - not persisting", flush=True)
    sys.exit(1)
if maxrho >= 0.5:
    print(f"MAX_LIB_RHO {maxrho} >= 0.5 - redundant vs library, not persisting", flush=True)
    sys.exit(1)

factor_id = "dd_pos_60"
doc = {
    "factor_id": factor_id,
    "factor_name": "60-day Range Position (drawdown-cycle phase)",
    "version": "1.0.0",
    "calculation": {
        "expression": "(close - rolling_min(close,60)) / (rolling_max(close,60) - rolling_min(close,60))",
        "description": ("Position of current close within its trailing 60-day high-low range "
                        "(min_periods=36). 1.0 = at/near the 60d high (recovered/breakout phase), "
                        "0.0 = at/near the 60d low (deep drawdown phase). Captures drawdown-cycle "
                        "phase and recovery strength across the 15-asset tradable cross-asset universe.")
    },
    "dependencies": ["close"],
    "parameters": {"window": 60, "min_periods": 36, "admission_horizon": 10},
    "expected_direction": 1,
    "validation": {
        "status": "EFFECTIVE",
        "period": "2020-01-01..2026-07-29",
        "admission_horizon": 10,
        "last_validated": "2026-07-30",
        "regime_notes": ("Cross-sectional rank IC vs 10d forward returns on the 15-asset tradable "
                         "universe. Regime split: 2020-2022 IC=0.0299 (ICIR=0.0904, n=431) positive "
                         "through COVID crash/recovery and 2022 bear; 2023-2024 IC=0.0061 (ICIR=0.0188, "
                         "n=305) flat in the AI-led equity rally / range regime; 2025-2026 IC=0.0687 "
                         "(ICIR=0.1946, n=234) strong. Non-monotonic across regimes but overall stable; "
                         "decay IC strengthens with horizon (0.0426 at 20d). Re-validate every ~3 months."),
        "metrics": m,
        "signal_artifact": build_artifact(sig),
    },
    "tags": ["trend", "drawdown", "recovery", "cross-asset", "range"],
}

with open(f"factors/{factor_id}.json", "w") as f:
    json.dump(doc, f, indent=1, default=str)
print(f"WROTE factors/{factor_id}.json", flush=True)

# ---- verify read-back ----
with open(f"factors/{factor_id}.json") as f:
    back = json.load(f)
assert back["factor_id"] == factor_id, "factor_id mismatch"
assert back["validation"]["status"] == "EFFECTIVE", "status mismatch"
assert back["validation"]["metrics"]["ic"] == m["ic"], "ic mismatch"
assert back["validation"]["metrics"]["icir"] == m["icir"], "icir mismatch"
assert back["validation"]["metrics"]["max_abs_library_correlation"] == round(maxrho, 3)
assert "signal_artifact" in back["validation"] and back["validation"]["signal_artifact"]["data"], "artifact missing"
print("READ-BACK OK: id=%s status=%s ic=%s icir=%s max_rho=%s artifact_bytes=%d" % (
    back["factor_id"], back["validation"]["status"], back["validation"]["metrics"]["ic"],
    back["validation"]["metrics"]["icir"],
    back["validation"]["metrics"]["max_abs_library_correlation"],
    len(back["validation"]["signal_artifact"]["data"])), flush=True)
