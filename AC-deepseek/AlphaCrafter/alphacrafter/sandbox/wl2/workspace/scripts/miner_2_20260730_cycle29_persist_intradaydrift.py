"""miner_2 cycle 29: persist intraday_drift_20 (EFFECTIVE) + signal artifact."""
import sys
sys.path.insert(0, "scripts")
import json
import numpy as np
import pandas as pd
from pathlib import Path
from miner2_lib import (load_close_panel, load_ohlc_panels, forward_returns,
                        validate_factor, load_library_signals, save_signal_artifact)

panel = load_close_panel()
ohlc = load_ohlc_panels()
lib = load_library_signals(panel)
fwd_cache = {str(h): forward_returns(panel, h) for h in (1, 2, 3, 5, 10, 20)}

open_p, close_p = ohlc["open"], ohlc["close"]
intraday = close_p / open_p - 1.0
factor = intraday.rolling(20, min_periods=10).mean()

m = validate_factor(factor, panel, library=lib, fwd_cache=fwd_cache)

factor_id = "intraday_drift_20"
doc = {
    "factor_id": factor_id,
    "factor_name": "Intraday Drift 20d (gap-free momentum)",
    "version": "1.0.0",
    "calculation": {
        "expression": "mean(close/open - 1, 20) with min_periods=10",
        "description": (
            "Per-asset mean of intraday (open->close) returns over the trailing 20 trading days "
            "on the asset's own calendar. Captures gap-free momentum: whether the asset tends to "
            "drift up or down during the trading session, stripping out overnight gap information "
            "that is already covered by close-based trend factors. Direction is raw; portfolio "
            "layer assigns sign from IC (positive in-sample)."
        )
    },
    "dependencies": ["open", "close"],
    "parameters": {"window": 20, "min_periods": 10},
    "validation": {
        "status": "EFFECTIVE",
        "validation_date": "2026-07-30",
        "period": {"start": "2020-01-01", "end": "2026-07-29"},
        "metrics": {
            "ic": m["ic"],
            "icir": m["icir"],
            "admission_horizon": 10,
            "ic_hit_ratio": m["ic_hit_ratio"],
            "n_ic_dates": m["n_ic_dates"],
            "coverage_asset_days": m["coverage_asset_days"],
            "coverage_dates_ge8": m["coverage_dates_ge8"],
            "n_dates_total": m["n_dates_total"],
            "n_dates_ge8": m["n_dates_ge8"],
            "turnover_10_rank": m["turnover_10_rank"],
            "decay_ic_by_horizon": m["decay_ic_by_horizon"],
            "max_abs_library_correlation": m["max_abs_library_correlation"],
            "library_pairwise_corr": m["library_pairwise_corr"],
            "corr_vs_effective_mom20_volproxy60": m["library_pairwise_corr"].get("mom20_volproxy60"),
            "corr_vs_effective_dxy_beta_cond_60x20": m["library_pairwise_corr"].get("dxy_beta_cond_60x20")
        },
        "gate": {"ic_threshold": 0.007, "icir_threshold": 0.084,
                 "correlation_threshold_effective_lib": 0.5},
        "regime_notes": (
            "Positive IC in 2020-21 (+0.084, ICIR 0.26), 2022 (+0.058, ICIR 0.17), and "
            "2025-26 (+0.042, ICIR 0.13); negative in 2023-24 (-0.028, ICIR -0.09) with clear "
            "recovery in the most recent regime. Full-sample IC=0.0353, ICIR=0.1073 pass the "
            "admission gate on 1677 IC dates across the 15-asset cross-asset universe. "
            "Correlation vs currently effective factors: mom20_volproxy60 rho=0.408, "
            "dxy_beta_cond_60x20 rho=0.045 (both < 0.5). max_abs_library_correlation=0.5002 "
            "arises vs quarantined mom_10d_skip5 only."
        )
    },
    "tags": ["momentum", "intraday", "price-structure", "cross-asset"],
    "last_validated": "2026-07-30T00:00:00"
}

path = Path("factors") / f"{factor_id}.json"
path.write_text(json.dumps(doc, indent=2, ensure_ascii=False))
print("wrote", path)

# Signal artifact so the post-Miner gate can recompute pairwise rho from real artifacts.
art = save_signal_artifact(factor_id, factor)
print("wrote", art, "shape", np.load(art).shape)

# ---- verify round-trip ----
back = json.loads(path.read_text())
assert back["factor_id"] == factor_id, "factor_id mismatch"
assert back["validation"]["status"] == "EFFECTIVE", "status not EFFECTIVE"
assert back["validation"]["metrics"]["ic"] == m["ic"]
assert back["validation"]["metrics"]["icir"] == m["icir"]
assert abs(back["validation"]["metrics"]["max_abs_library_correlation"]) > 0
# artifact reload matches the factor panel used for validation
arr = np.load(art)
assert arr.shape == factor.values.shape
assert np.allclose(arr, factor.values, equal_nan=True), "artifact does not match factor panel"
print("ROUND-TRIP OK: id=%s status=%s ic=%.4f icir=%.4f maxlibcorr=%.4f art=%s" % (
    back["factor_id"], back["validation"]["status"], back["validation"]["metrics"]["ic"],
    back["validation"]["metrics"]["icir"], back["validation"]["metrics"]["max_abs_library_correlation"], art))
