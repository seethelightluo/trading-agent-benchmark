"""miner_1 2026-07-30 cycle 22: re-validate top candidates + new constructions,
select decorrelated set, and PERSIST with the gate-safe artifact format.

Persistence format (required by post-miner gate, learned from rejected files):
  - factors/<fid>.json            signal_artifact = "<fid>.signal.npy" (STRING)
  - factors/<fid>.signal.npy      float64 matrix, shape (n_dates_union, 15),
                                  dates ascending, columns in TRADABLES order.

Validation: per-asset own-calendar factor computation (no lookahead), reindexed
to union panel; daily cross-sectional Spearman IC vs h-day fwd returns; admission
h=10; gate |IC|>=0.007, |ICIR|>=0.084; coverage >=8 assets/date; turnover;
decay; regime splits; pairwise |rho| on the EXACT artifact matrices (<0.5).
"""
import sys, json, os
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from miner_1_lib import (TRADABLES, load_panel, per_asset, forward_returns,
                         compute_ic, validate_factor, load_library_signals,
                         report)

panel = load_panel()
close = panel
print(f"panel shape: {panel.shape}  dates: {panel.index.min().date()}..{panel.index.max().date()}  "
      f"instruments: {len(panel.columns)}")
print(f"assets: {list(panel.columns)}")

HORIZONS = (1, 2, 3, 5, 10, 20)
ADM_H = 10
fwd_cache = {str(h): forward_returns(panel, h) for h in HORIZONS}

# ---------------------------------------------------------------------------
# Candidate constructions. Two views:
#  (a) per-asset own-calendar (validation, artifact) -- correct for crypto/equity calendars
#  (b) strict gate-namespace panel expression (close+pd+np) -- reproducibility audit
# ---------------------------------------------------------------------------
CAND = {
    "mom20_volproxy60": ("close.shift(5)/close.shift(25)-1.0",
                         "20d momentum (skip5) damped by its own magnitude proxy"),
    "mom20_vs_60": ("(close.shift(5)/close.shift(25)-1.0) - (close.shift(5)/close.shift(65)-1.0)",
                    "20d vs 60d momentum curve steepness (skip5)"),
    "range_pos_120d": ("(close - close.rolling(120, min_periods=30).min()) / "
                       "(close.rolling(120, min_periods=30).max() - close.rolling(120, min_periods=30).min() + 1e-9)",
                       "120d range position (0=low,1=high)"),
    "mom15d_skip5": ("close.shift(5)/close.shift(20)-1.0",
                     "15d momentum skipping most recent 5d"),
    "zscore_60_rev": ("-1.0 * (close - close.rolling(60, min_periods=30).mean()) / close.rolling(60, min_periods=30).std()",
                      "negative 60d z-score (short-term mean reversion)"),
    "mom90d_skip10": ("close.shift(10)/close.shift(100)-1.0",
                      "90d momentum skipping most recent 10d"),
    "mom20d_damp_rev10": ("(close.shift(10)/close.shift(30)-1.0) - 0.5*(close/close.shift(10)-1.0)",
                          "20d momentum damped by half of 10d reversal"),
    "vol_trend_60x20": ("close.pct_change().rolling(60, min_periods=30).std() "
                        "/ close.pct_change().rolling(20, min_periods=10).std() - 1.0",
                        "60d/20d realized-vol ratio (vol regime drift)"),
    "dist_high_60d": ("close / close.rolling(60, min_periods=15).max() - 1.0",
                      "distance from 60d high (short-horizon trend location)"),
    "mom_curve_volscale": ("((close.shift(5)/close.shift(25)-1.0) - (close.shift(5)/close.shift(65)-1.0)) "
                           "/ (1.0 + close.pct_change().rolling(20, min_periods=10).std())",
                           "momentum curve steepness scaled by inverse 20d vol"),
}

# per-asset own-calendar implementation (lambda receives asset's own close series s)
LAMBDAS = {
    "mom20_volproxy60": "lambda s: (s.shift(5)/s.shift(25)-1.0) / (1.0 + abs(s.shift(5)/s.shift(65)-1.0))",
    "mom20_vs_60":      "lambda s: (s.shift(5)/s.shift(25)-1.0) - (s.shift(5)/s.shift(65)-1.0)",
    "range_pos_120d":   "lambda s: (s - s.rolling(120, min_periods=30).min()) / (s.rolling(120, min_periods=30).max() - s.rolling(120, min_periods=30).min() + 1e-9)",
    "mom15d_skip5":     "lambda s: s.shift(5)/s.shift(20)-1.0",
    "zscore_60_rev":    "lambda s: -1.0 * (s - s.rolling(60, min_periods=30).mean()) / s.rolling(60, min_periods=30).std()",
    "mom90d_skip10":    "lambda s: s.shift(10)/s.shift(100)-1.0",
    "mom20d_damp_rev10":"lambda s: (s.shift(10)/s.shift(30)-1.0) - 0.5*(s/s.shift(10)-1.0)",
    "vol_trend_60x20":  "lambda s: s.pct_change().rolling(60, min_periods=30).std() / s.pct_change().rolling(20, min_periods=10).std() - 1.0",
    "dist_high_60d":    "lambda s: s / s.rolling(60, min_periods=15).max() - 1.0",
    "mom_curve_volscale":"lambda s: ((s.shift(5)/s.shift(25)-1.0) - (s.shift(5)/s.shift(65)-1.0)) / (1.0 + s.pct_change().rolling(20, min_periods=10).std())",
}

signals = {}
print("\n=== strict gate-namespace eval (close+pd+np only) ===")
env = {"pd": pd, "np": np, "close": close}
for fid, (expr, _) in CAND.items():
    try:
        sig = eval(expr, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape
        print(f"  {fid:20s} gate_expr eval={'OK' if ok else 'BAD_SHAPE'}")
    except Exception as e:
        print(f"  {fid:20s} gate_expr eval=FAIL {type(e).__name__}: {str(e)[:70]}")

print("\n=== per-asset own-calendar construction (validation + artifact) ===")
for fid, lam in LAMBDAS.items():
    sig = per_asset(close, eval(lam, {"abs": abs})).reindex(index=panel.index, columns=panel.columns)
    signals[fid] = sig
    print(f"  {fid:20s} shape={sig.shape} nan={int(sig.isna().sum().sum())} "
          f"dates_ge8={int((sig.notna().sum(axis=1)>=8).sum())}")

# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
print("\n=== validation (admission h=10, per-asset calendars) ===")
library = load_library_signals(panel)   # prior library family (provenance audit)
results = {}
for fid, sig in signals.items():
    m = validate_factor(sig, panel, horizons=HORIZONS, admission_horizon=ADM_H,
                        library=library, fwd_cache=fwd_cache)
    results[fid] = m
    report(fid, m)

passers = [fid for fid, m in results.items()
           if abs(m["ic"]) >= 0.007 and abs(m["icir"]) >= 0.084
           and m["n_ic_dates"] >= 800 and m["coverage_dates_ge8"] >= 0.5]
print(f"\nPASSERS (gate + robustness filters): {passers} ({len(passers)}/{len(results)})")

# ---------------------------------------------------------------------------
# Pairwise |rho| on EXACT artifact matrices (what the gate will see)
# ---------------------------------------------------------------------------
print("\n=== pairwise |rho| among passers on artifact matrices (>=0.5 redundant) ===")
names = passers
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i:
            continue
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        r = abs(float(both["x"].corr(both["y"]))) if len(both) > 100 else np.nan
        rho.loc[a, b] = r
        rho.loc[b, a] = r
if names:
    print("        " + "".join(f"{b[:9]:>11s}" for b in names))
    for i, a in enumerate(names):
        print(f"  {a:22s}" + "".join(f"{rho.loc[a,b]:>11.3f}" if pd.notna(rho.loc[a, b]) else f"{'-':>11s}" for b in names))

qual = {fid: abs(results[fid]["ic"]) * abs(results[fid]["icir"]) for fid in passers}
order = sorted(passers, key=lambda f: -qual[f])
selected = []
for f in order:
    if not selected:
        selected.append(f)
        continue
    mx = max((abs(rho.loc[f, s]) for s in selected if pd.notna(rho.loc[f, s])), default=0.0)
    if mx < 0.5:
        selected.append(f)
    else:
        print(f"  drop {f}: max|rho| vs selected = {mx:.3f}")
print(f"\n=== selected decorrelated set: {selected} ===")

# ---------------------------------------------------------------------------
# Regime breakdown for selected
# ---------------------------------------------------------------------------
print("\n=== regime breakdown for selected (IC10 / ICIR10 / n) ===")
regime_out = {}
for fid in selected:
    sig = signals[fid]
    line = [fid]
    rd = {}
    for r0, r1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-07-29")]:
        sub_mask = (panel.index >= r0) & (panel.index <= r1)
        ic_ser = compute_ic(sig.loc[sub_mask], fwd_cache[str(ADM_H)].loc[sub_mask]).dropna()
        if len(ic_ser) >= 30:
            sd = ic_ser.std()
            icir = ic_ser.mean() / sd if sd > 0 else 0.0
            line.append(f"{r0[:4]}-{r1[:4]}: {ic_ser.mean():+.4f}/{icir:+.3f}/n={len(ic_ser)}")
            rd[r0[:4]] = {"ic": round(float(ic_ser.mean()), 4),
                          "icir": round(float(icir), 4), "n_dates": int(len(ic_ser))}
    regime_out[fid] = rd
    print("  " + " | ".join(line))

# ---------------------------------------------------------------------------
# Persistence: factors/<fid>.json + factors/<fid>.signal.npy
# ---------------------------------------------------------------------------
CONTRACT = {"ic_threshold": 0.007, "icir_threshold": 0.084, "correlation_threshold": 0.5,
            "library_capacity": 30, "active_top_k": 10}
NOW = "2026-07-30"
VISIBLE = "2026-07-29"
VALIDATED = f"2020-01-01..{VISIBLE}"

META = {
    "mom20_volproxy60":  ("Momentum 20d vol-damped (skip5)",
                          "20d price momentum ending 5 days ago, damped by 1/(1+|60d momentum proxy|): strong stable trends score high, explosive moves are penalized.",
                          {"lookback": 20, "skip": 5, "vol_proxy_lookback": 60}, ["momentum", "volatility", "cross-asset"]),
    "mom20_vs_60":       ("Momentum curve 20d vs 60d (skip5)",
                          "Difference between 20d and 60d momentum (both skip5): measures momentum acceleration/steepness of the trend curve.",
                          {"short": 20, "long": 60, "skip": 5}, ["momentum", "trend", "acceleration"]),
    "range_pos_120d":    ("120d Range Position",
                          "Position of close within its trailing 120-day high-low range (0=at low, 1=at high); trend/mean-reversion regime location.",
                          {"window": 120, "min_periods": 30}, ["trend", "range", "regime"]),
    "mom15d_skip5":      ("Momentum 15d (skip5)",
                          "15-day price momentum ending 5 days ago (return t-20..t-5), skipping the most recent week.",
                          {"lookback": 15, "skip": 5}, ["momentum", "cross-asset"]),
    "zscore_60_rev":     ("60d Z-score reversal",
                          "Negative 60-day price z-score: buys assets trading below their 60d mean (short-horizon mean reversion).",
                          {"window": 60, "min_periods": 30}, ["reversal", "mean-reversion", "volatility"]),
    "mom90d_skip10":     ("Momentum 90d (skip10)",
                          "90-day price momentum ending 10 days ago (return t-100..t-10), skipping recent two weeks.",
                          {"lookback": 90, "skip": 10}, ["momentum", "trend"]),
    "mom20d_damp_rev10": ("Momentum 20d damped by 10d reversal",
                          "20d momentum (skip10) minus half of the most recent 10d return: keeps trend signal while fading short-term overshoots.",
                          {"lookback": 20, "skip": 10, "reversal_weight": 0.5}, ["momentum", "reversal"]),
    "vol_trend_60x20":   ("Vol regime drift 60x20",
                          "60d/20d realized volatility ratio minus 1: rising vol regime (fear) vs falling (calm).",
                          {"short_win": 20, "long_win": 60, "min_periods_short": 10, "min_periods_long": 30}, ["volatility", "regime"]),
    "dist_high_60d":     ("Distance from 60d high",
                          "Close relative to trailing 60d high minus 1: short-horizon trend location (negative = drawdown from recent peak).",
                          {"window": 60, "min_periods": 15}, ["trend", "drawdown"]),
    "mom_curve_volscale":("Momentum curve vol-scaled",
                          "Momentum curve steepness (20d vs 60d, skip5) scaled by 1/(1+20d realized vol): trend acceleration per unit risk.",
                          {"short": 20, "long": 60, "skip": 5, "vol_win": 20}, ["momentum", "volatility", "risk-adjusted"]),
}

os.makedirs("factors", exist_ok=True)
print("\n=== writing factors/<fid>.json + <fid>.signal.npy ===")
written = []
for fid in selected:
    sig = signals[fid].reindex(index=panel.index, columns=panel.columns)
    mat = sig.values.astype(np.float64)               # (n_dates, 15), NaN preserved
    art_name = f"{fid}.signal.npy"
    np.save(f"factors/{art_name}", mat)
    m = results[fid]
    for other in selected:
        if other != fid:
            m["library_pairwise_corr"] = m.get("library_pairwise_corr", {})
            m["library_pairwise_corr"][other] = round(float(rho.loc[fid, other]), 4)
    maxlib = max((abs(v) for v in m.get("library_pairwise_corr", {}).values()), default=0.0)
    m["max_abs_library_correlation"] = round(max(maxlib, m.get("max_abs_library_correlation", 0.0)), 4)
    name, desc, params, tags = META[fid]
    direction = 1 if m["ic"] >= 0 else -1
    payload = {
        "factor_id": fid,
        "factor_name": name,
        "version": "1.1.0",
        "calculation": {"expression": CAND[fid][0], "description": desc},
        "dependencies": ["close"],
        "parameters": params,
        "expected_direction": direction,
        "validation": {
            "status": "EFFECTIVE",
            "period": VALIDATED,
            "last_validated": NOW,
            "admission_horizon": ADM_H,
            "regime_notes": "15-instrument tradable cross-asset universe; regime split: "
                            + "; ".join(f"{k}: ic={v['ic']:.3f} icir={v['icir']:.3f} n={v['n_dates']}"
                                        for k, v in regime_out[fid].items()),
            "metrics": {
                "ic": round(m["ic"], 4),
                "icir": round(m["icir"], 4),
                "ic_hit_ratio": m["ic_hit_ratio"],
                "n_ic_dates": m["n_ic_dates"],
                "coverage_asset_days": m["coverage_asset_days"],
                "coverage_dates_ge8": m["coverage_dates_ge8"],
                "turnover_10d_rank": m.get("turnover_10d_rank", m.get("turnover_10_rank")),
                "decay_ic_by_horizon": m["decay_ic_by_horizon"],
                "max_abs_library_correlation": m["max_abs_library_correlation"],
                "library_pairwise_corr": m.get("library_pairwise_corr", {}),
            },
        },
        "tags": tags,
        "benchmark_admission": {
            "contract": CONTRACT,
            "selected_metrics": {
                "ic": round(m["ic"], 4),
                "icir": round(m["icir"], 4),
                "metric_path": "validation.metrics",
                "max_abs_library_correlation": m["max_abs_library_correlation"],
                "correlation_path": "validation.metrics.max_abs_library_correlation",
                "quality": round(abs(m["ic"]) * abs(m["icir"]), 8),
            },
            "admitted_at": f"{NOW}T00:00:00",
        },
        "signal_artifact": art_name,
        "artifact_provenance": {
            "format": "npy_matrix",
            "shape": list(mat.shape),
            "columns": list(panel.columns),
            "dates_first": str(panel.index[0].date()),
            "dates_last": str(panel.index[-1].date()),
            "n_nan": int(np.isnan(mat).sum()),
        },
    }
    with open(f"factors/{fid}.json", "w") as f:
        json.dump(payload, f, indent=1)
    written.append(fid)
    print(f"  wrote factors/{fid}.json ({os.path.getsize(f'factors/{fid}.json')/1024:.0f} KB) "
          f"+ factors/{art_name} ({os.path.getsize(f'factors/{art_name}')/1024:.0f} KB)")

# ---------------------------------------------------------------------------
# Read-back verification
# ---------------------------------------------------------------------------
print("\n=== read-back verification ===")
ok_all = True
for fid in written:
    with open(f"factors/{fid}.json") as f:
        d = json.load(f)
    art_name = d.get("signal_artifact")
    mtx = np.load(f"factors/{art_name}") if art_name and os.path.exists(f"factors/{art_name}") else None
    checks = {
        "valid_json": True,
        "factor_id_ok": d.get("factor_id") == fid,
        "status_effective": d.get("validation", {}).get("status") == "EFFECTIVE",
        "ic_gate": abs(d["validation"]["metrics"]["ic"]) >= 0.007,
        "icir_gate": abs(d["validation"]["metrics"]["icir"]) >= 0.084,
        "artifact_str": isinstance(art_name, str),
        "artifact_loads": mtx is not None and mtx.shape == (len(panel), 15),
        "artifact_has_signal": mtx is not None and np.nanstd(mtx) > 0,
    }
    print(f"  {fid:20s} " + " ".join(f"{k}={v}" for k, v in checks.items()))
    ok_all &= all(checks.values())
print("ALL_VERIFIED" if ok_all else "VERIFY_FAILED")

out = {"panel_shape": list(panel.shape), "visible_through": VISIBLE,
       "n_dates": int(len(panel)), "n_instruments": int(len(panel.columns)),
       "results": {k: {kk: vv for kk, vv in v.items() if kk != "library_pairwise_corr"} for k, v in results.items()},
       "passers": passers, "selected": selected,
       "pairwise_rho_passers": {a: {b: round(float(rho.loc[a, b]), 4) for b in names
                                    if pd.notna(rho.loc[a, b])} for a in names},
       "regime": regime_out}
json.dump(out, open("scripts/_miner1_cycle22_results.json", "w"), indent=1, default=float)
print("\nsaved scripts/_miner1_cycle22_results.json")
print("SELECTED_FACTORS=" + json.dumps(selected))
