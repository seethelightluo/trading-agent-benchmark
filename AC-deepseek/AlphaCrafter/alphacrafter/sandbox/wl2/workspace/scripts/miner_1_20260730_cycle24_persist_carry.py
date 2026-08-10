"""miner_1 2026-07-30 cycle 24: persist carry/term-structure factors that passed
the admission gate in cycle 23 exploration.

Deterministic recompute from raw data; write JSON + NPY signal artifacts;
read-back verification. No fabricated metrics.
"""
import sys, json
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (load_panel, per_asset, forward_returns, compute_ic,
                         validate_factor, load_library_signals, panel_rank_corr,
                         coverage_stats)

panel = load_panel()
HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

CAND = {
    "carry_12m3m": ("lambda s: (s.shift(63)/s.shift(252)-1.0) - (s/s.shift(63)-1.0)",
                    "12m-3m carry proxy: 12m return (ex last quarter) minus last-quarter return",
                    "close.shift(63)/close.shift(252)-1.0 - (close/close.shift(63)-1.0)",
                    {"window_long": 252, "window_short": 63, "skip": 0}),
    "carry_3m1m":  ("lambda s: (s.shift(21)/s.shift(63)-1.0) - (s/s.shift(21)-1.0)",
                    "3m-1m carry proxy: 3m return (ex last month) minus last-month return",
                    "close.shift(21)/close.shift(63)-1.0 - (close/close.shift(21)-1.0)",
                    {"window_long": 63, "window_short": 21, "skip": 0}),
}
DESC = {k: v[1] for k, v in CAND.items()}
EXPR = {k: v[2] for k, v in CAND.items()}
PARAM = {k: v[3] for k, v in CAND.items()}

# --- recompute signals ---
signals = {}
for fid, (lam, *_rest) in CAND.items():
    sig = per_asset(panel, eval(lam)).reindex(index=panel.index, columns=panel.columns)
    signals[fid] = sig
    print(f"recomputed {fid}: shape={sig.shape} nan={int(sig.isna().sum().sum())} "
          f"dates_ge8={int((sig.notna().sum(axis=1)>=8).sum())}")

# --- library: older signals recomputed + effective factors loaded from NPY artifacts ---
library = load_library_signals(panel)
for fid in ["mom20_volproxy60", "mom_curve_volscale", "range_pos_120d"]:
    arr = np.load(f"factors/{fid}.signal.npy")
    df = pd.DataFrame(arr, index=panel.index, columns=panel.columns)
    library[fid] = df
    print(f"library loaded from artifact: {fid} shape={df.shape}")

# --- validation ---
print("\n=== validation (admission h=10, gates |IC|>=0.007 |ICIR|>=0.084) ===")
results = {}
for fid, sig in signals.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=library, fwd_cache=fwd_cache)
    results[fid] = m
    passed = abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
    print(f"[{fid}] IC={m['ic']} ICIR={m['icir']} hit={m['ic_hit_ratio']} "
          f"n={m['n_ic_dates']} cov_asset={m['coverage_asset_days']} "
          f"cov_dates={m['coverage_dates_ge8']} turnover10={m['turnover_10_rank']} "
          f"maxlibcorr={m['max_abs_library_correlation']} => {'PASS' if passed else 'FAIL'}")

# pairwise correlation between the two carry factors
rho_12_3 = panel_rank_corr(signals["carry_12m3m"], signals["carry_3m1m"])
print(f"\npairwise rank rho carry_12m3m vs carry_3m1m = {rho_12_3:.4f}")

# regime breakdown
print("\n=== regime IC10/ICIR10 ===")
regime_out = {}
for fid in CAND:
    rd = {}
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub = (panel.index >= r0) & (panel.index <= r1)
        ic_ser = compute_ic(signals[fid].loc[sub], fwd_cache[str(ADM_H)].loc[sub]).dropna()
        if len(ic_ser) >= 30:
            sd = ic_ser.std()
            icir = ic_ser.mean() / sd if sd > 0 else 0.0
            rd[r0[:4]] = {"ic": round(float(ic_ser.mean()), 4),
                          "icir": round(float(icir), 4), "n_dates": int(len(ic_ser))}
    regime_out[fid] = rd
    print(f"  {fid}: " + " | ".join(f"{k}: {v['ic']:+.4f}/{v['icir']:+.3f}/n={v['n_dates']}"
                                    for k, v in rd.items()))

# --- persist passers ---
PASSERS = [fid for fid, m in results.items()
           if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
           and m["n_ic_dates"] >= 800 and m["coverage_dates_ge8"] >= 0.5]
print(f"\nPASSERS: {PASSERS}")

TS = "2026-07-30T00:00:00"
for fid in PASSERS:
    m = results[fid]
    direction = 1 if m["ic"] > 0 else -1
    cov = coverage_stats(signals[fid])
    sig = signals[fid]
    np.save(f"factors/{fid}.signal.npy", sig.astype(np.float64).values)
    reg_notes = "; ".join(f"{k}: ic={v['ic']} icir={v['icir']} n={v['n_dates']}"
                          for k, v in regime_out[fid].items())
    quality = round(abs(m["ic"]) * abs(m["icir"]), 8)
    doc = {
        "factor_id": fid,
        "factor_name": f"Cross-asset carry proxy {fid}",
        "version": "1.0.0",
        "calculation": {
            "expression": EXPR[fid],
            "description": DESC[fid],
        },
        "dependencies": ["close"],
        "parameters": PARAM[fid],
        "expected_direction": direction,
        "validation": {
            "status": "EFFECTIVE",
            "period": "2020-01-01..2026-07-29",
            "last_validated": "2026-07-30",
            "admission_horizon": ADM_H,
            "regime_notes": "15-instrument tradable cross-asset universe; " + reg_notes,
            "metrics": {
                "ic": m["ic"],
                "icir": m["icir"],
                "ic_hit_ratio": m["ic_hit_ratio"],
                "n_ic_dates": m["n_ic_dates"],
                "coverage_asset_days": m["coverage_asset_days"],
                "coverage_dates_ge8": m["coverage_dates_ge8"],
                "turnover_10d_rank": m["turnover_10_rank"],
                "decay_ic_by_horizon": m["decay_ic_by_horizon"],
                "max_abs_library_correlation": m["max_abs_library_correlation"],
                "library_pairwise_corr": m["library_pairwise_corr"],
            },
        },
        "tags": ["carry", "term-structure", "cross-asset"],
        "benchmark_admission": {
            "contract": {"ic_threshold": 0.007, "icir_threshold": 0.084,
                         "correlation_threshold": 0.5, "library_capacity": 30,
                         "active_top_k": 10},
            "selected_metrics": {"ic": m["ic"], "icir": m["icir"],
                                 "metric_path": "validation.metrics",
                                 "max_abs_library_correlation": m["max_abs_library_correlation"],
                                 "correlation_path": "validation.metrics.max_abs_library_correlation",
                                 "quality": quality},
            "admitted_at": TS,
        },
        "signal_artifact": f"{fid}.signal.npy",
        "artifact_provenance": {
            "format": "npy_matrix",
            "shape": list(sig.shape),
            "columns": list(panel.columns),
            "dates_first": str(panel.index.min().date()),
            "dates_last": str(panel.index.max().date()),
            "n_nan": int(sig.isna().sum().sum()),
        },
    }
    with open(f"factors/{fid}.json", "w") as f:
        json.dump(doc, f, indent=1)
    print(f"persisted factors/{fid}.json + {fid}.signal.npy (direction={direction})")

# --- read-back verification ---
print("\n=== read-back verification ===")
ok = True
for fid in PASSERS:
    with open(f"factors/{fid}.json") as f:
        d = json.load(f)
    arr = np.load(f"factors/{fid}.signal.npy")
    checks = {
        "valid_json": True,
        "factor_id": d["factor_id"] == fid,
        "status": d["validation"]["status"] == "EFFECTIVE",
        "ic_threshold": abs(d["validation"]["metrics"]["ic"]) >= 0.007,
        "icir_threshold": abs(d["validation"]["metrics"]["icir"]) >= 0.084,
        "artifact_shape": arr.shape == tuple(d["artifact_provenance"]["shape"]) == (2398, 15),
        "artifact_reloadable": arr.shape[0] == len(panel),
        "has_signal_provenance": "signal_artifact" in d and "artifact_provenance" in d,
    }
    for k, v in checks.items():
        print(f"  {fid} {k}: {'OK' if v else 'FAIL'}")
        ok = ok and v
print("ALL READ-BACK CHECKS:", "PASS" if ok else "FAIL")

json.dump({"persisted": PASSERS, "rho_12m3m_vs_3m1m": round(rho_12_3, 4),
           "results": {k: {kk: vv for kk, vv in v.items() if kk != "library_pairwise_corr"}
                       for k, v in results.items()}},
          open("scripts/_miner1_cycle24_persist_results.json", "w"), indent=1, default=float)
print("DONE")
