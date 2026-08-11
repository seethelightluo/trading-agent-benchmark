"""miner_3 2026-09-10: persist the 5 IC/ICIR-gate passing candidates as EFFECTIVE.

Candidates (validated on data visible through 2026-09-09):
  updown_vol_ratio_20 (+), downside_freq_20 (-), max_gain_20 (+),
  cn10y_corr_60 (-), hl_rank_20 (+)
Signal npy artifacts already saved by validate_passing.py. Here we recompute
full metrics (hit ratio, coverage, turnover, decay) from the artifacts and write
factors/<fid>.json, then read back and verify.
"""
import sys, json, glob, os, datetime
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
from miner_3_20260813_lib import (ASSETS, GRID, HORIZON, to_grid,
                                  cross_sectional_rank, spearman_ic_matrix,
                                  summarize, decay_curve, fwd_by_horizon_dict,
                                  turnover_10d_rank, coverage_stats, MIN_ASSETS)

GATE_IC = 0.0070
GATE_ICIR = 0.0840
CORR_LIMIT = 0.5

# validation results from validate_passing.py (IC/ICIR/regime/decay recomputed there)
vres = json.load(open("scripts/miner_3_20260910_validate_passing_results.json"))

def load_asset(sym, days=2300):
    df = get_stock_daily_data(sym, days=days)
    if df is None:
        return None
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df = df.set_index("date")
    for c in ["open", "close", "high", "low", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["close"] = df["close"].astype(float)
    return df

series = {s: load_asset(s) for s in ASSETS}
series = {s: df for s, df in series.items() if df is not None and len(df) > 100}
fwd = fwd_by_horizon_dict(series)
dates = np.array(GRID)
print(f"grid rows={len(GRID)} first={GRID[0]} last={GRID[-1]} assets={len(series)}/15")

# kept-library membership for conflict labelling
KEPT = set()
for f in sorted(glob.glob("factors/*.json")):
    try:
        d = json.load(open(f))
        if d.get("validation", {}).get("status") == "EFFECTIVE" and \
           os.path.exists(f"factors/{d.get('factor_id','')}.signal.npy"):
            KEPT.add(d.get("factor_id"))
    except Exception:
        pass

CONTRACT = {"ic_threshold": GATE_IC, "icir_threshold": GATE_ICIR,
            "correlation_threshold": CORR_LIMIT, "library_capacity": 30,
            "active_top_k": 10}

META = {
 "updown_vol_ratio_20": {
   "name": "Up/Down volatility ratio 20d",
   "expression": "rolling_std(max(daily_ret,0),20) / rolling_std(min(daily_ret,0),20)",
   "description": ("Ratio of the 20d std of positive daily returns to the 20d std of negative daily returns "
                   "(own-calendar). High values indicate upside moves are more volatile than downside moves "
                   "(positive-skewed tape); low values indicate downside risk dominates. Positive cross-sectional "
                   "predictor of forward 10d returns in cross-asset universe."),
   "dependencies": ["close"],
   "parameters": {"window": 20, "min_periods": 8},
   "direction": 1,
   "tags": ["volatility", "skew", "asymmetry", "cross-asset"]},
 "downside_freq_20": {
   "name": "Downside frequency 20d",
   "expression": "rolling_mean(daily_ret < 0, 20)",
   "description": ("Fraction of trailing 20 trading days with negative returns. High values identify assets in "
                   "persistent drawdown tapes; low values identify assets with predominantly positive days. "
                   "Negative cross-sectional predictor of forward 10d returns (frequent downside days forecast "
                   "relative weakness)."),
   "dependencies": ["close"],
   "parameters": {"window": 20, "min_periods": 10},
   "direction": -1,
   "tags": ["trend", "momentum", "regime", "cross-asset"]},
 "max_gain_20": {
   "name": "Max single-day gain 20d",
   "expression": "rolling_max(daily_ret, 20)",
   "description": ("Largest single-day return over the trailing 20 trading days. Identifies assets with recent "
                   "explosive upside participation (liquidity/attention proxy). Positive cross-sectional predictor "
                   "of forward 10d returns, strongest in 2020-2022 and 2025-2026 regimes."),
   "dependencies": ["close"],
   "parameters": {"window": 20, "min_periods": 10},
   "direction": 1,
   "tags": ["momentum", "extremes", "cross-asset"]},
 "cn10y_corr_60": {
   "name": "CN10Y yield correlation 60d",
   "expression": "rolling_corr(daily_ret_asset, daily_ret_CN10Y, 60)",
   "description": ("60d rolling correlation between asset daily returns and CN10Y yield daily changes. Assets "
                   "positively co-moving with Chinese 10y yields behave like growth/risk proxies; negative IC means "
                   "higher co-movement with CN10Y forecasts relative underperformance (yield-sensitive crowding). "
                   "Negative cross-sectional predictor of forward 10d returns."),
   "dependencies": ["close", "CN10Y.close"],
   "parameters": {"window": 60, "min_periods": 40},
   "direction": -1,
   "tags": ["cross-asset", "correlation", "rates", "macro"]},
 "hl_rank_20": {
   "name": "High-Low range position 20d",
   "expression": "(close - rolling_min(low,20)) / (rolling_max(high,20) - rolling_min(low,20))",
   "description": ("Position of current close within the trailing 20d high-low range (0..1). High values indicate "
                   "price near the top of the recent range (breakout proximity); low values near the bottom. "
                   "Positive cross-sectional predictor of forward 10d returns (range-momentum continuation)."),
   "dependencies": ["close", "high", "low"],
   "parameters": {"window": 20, "min_periods": 10},
   "direction": 1,
   "tags": ["momentum", "range", "technical", "cross-asset"]},
}

def build(fid, meta):
    v = vres[fid]
    mat = np.load(f"factors/{fid}.signal.npy", allow_pickle=True)
    rank_mat = cross_sectional_rank(mat)
    summ = summarize(spearman_ic_matrix(mat, fwd[HORIZON]), dates, fid, HORIZON)
    cov_ad, cov_d8 = coverage_stats(mat)
    to = turnover_10d_rank(rank_mat)
    dec = decay_curve(mat, fwd)
    ic, icir = summ["ic"], summ["icir"]
    q = abs(ic) * abs(icir)
    n_nan = int(np.isnan(mat).sum())
    corrs = {name: rho for name, rho in v["conflicts"]}
    # library pairwise corr snapshot: keep conflicts plus max-lib pair
    lib_snapshot = {}
    if v.get("max_lib_corr_name"):
        lib_snapshot[v["max_lib_corr_name"]] = round(v["max_abs_library_correlation"], 4)
    for name, rho in v["conflicts"]:
        if name not in lib_snapshot:
            lib_snapshot[name] = round(rho, 4)
    reg_notes = "; ".join(f"{k}: ic={v['regime'][k]['ic']:.4f} icir={v['regime'][k]['icir']:.3f} n={v['regime'][k]['n']}"
                          for k in v["regime"])
    conflicts_kept = [c for c in v["conflicts"] if c[0] in KEPT and abs(c[1]) >= CORR_LIMIT]
    doc = {
        "factor_id": fid,
        "factor_name": meta["name"],
        "version": "1.0.0",
        "calculation": {
            "expression": meta["expression"],
            "description": meta["description"]},
        "dependencies": meta["dependencies"],
        "parameters": meta["parameters"],
        "expected_direction": meta["direction"],
        "validation": {
            "status": "EFFECTIVE",
            "period": f"{GRID[0]}..{GRID[-1]}",
            "last_validated": "2026-09-10",
            "admission_horizon": HORIZON,
            "regime_notes": (f"15-instrument tradable cross-asset universe; data visible through {GRID[-1]}. "
                             f"{reg_notes}. KEPT-library conflicts>|0.5|: {conflicts_kept or 'none'}"),
            "metrics": {
                "ic": round(ic, 4),
                "icir": round(icir, 4),
                "ic_hit_ratio": round(summ["hit"], 4),
                "n_ic_dates": int(summ["n_ic_dates"]),
                "coverage_asset_days": round(cov_ad, 4),
                "coverage_dates_ge8": round(cov_d8, 4),
                "turnover_10d_rank": round(to, 4),
                "decay_ic_by_horizon": dec,
                "max_abs_library_correlation": round(v["max_abs_library_correlation"], 4),
                "library_pairwise_corr": lib_snapshot}},
        "tags": meta["tags"],
        "benchmark_admission": {
            "contract": CONTRACT,
            "selected_metrics": {
                "ic": round(ic, 4), "icir": round(icir, 4),
                "metric_path": "validation.metrics",
                "reported_max_abs_library_correlation": round(v["max_abs_library_correlation"], 4),
                "correlation_path": "validation.metrics.max_abs_library_correlation",
                "quality": round(q, 8)},
            "admitted_at": datetime.datetime.now(datetime.timezone.utc).isoformat()},
        "signal_artifact": f"{fid}.signal.npy",
        "artifact_provenance": {
            "format": "npy_matrix",
            "shape": list(mat.shape),
            "columns": ASSETS,
            "dates_first": GRID[0],
            "dates_last": GRID[-1],
            "n_nan": n_nan}}
    return doc

os.makedirs("factors", exist_ok=True)
for fid, meta in META.items():
    doc = build(fid, meta)
    path = f"factors/{fid}.json"
    json.dump(doc, open(path, "w"), indent=1)
    print(f"WROTE {path}")

print("=== READBACK VERIFICATION ===")
ok_all = True
for fid in META:
    path = f"factors/{fid}.json"
    d = json.load(open(path))
    v = d["validation"]
    m = v["metrics"]
    checks = {
        "id_match": d["factor_id"] == fid,
        "status": v["status"] == "EFFECTIVE",
        "ic_gate": abs(m["ic"]) >= GATE_IC,
        "icir_gate": abs(m["icir"]) >= GATE_ICIR,
        "signal_exists": os.path.exists(f"factors/{fid}.signal.npy"),
        "artifact_ref": d["signal_artifact"] == f"{fid}.signal.npy",
        "maxlibcorr_reported": "max_abs_library_correlation" in m,
        "last_validated": d["validation"]["last_validated"] == "2026-09-10",
    }
    ok = all(checks.values())
    ok_all &= ok
    print(f"{fid}: {checks} -> {'OK' if ok else 'FAIL'}")
print("ALL OK" if ok_all else "SOME FAILED")
