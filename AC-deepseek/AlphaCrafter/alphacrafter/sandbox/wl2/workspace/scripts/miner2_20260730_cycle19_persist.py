"""miner_2 2026-07-30: persist cycle-19 passing factors with embedded signal artifacts.

Context: the post-Miner gate quarantines any factors/*.json without a recoverable
signal artifact (it recomputes pairwise rho from real artifacts, never assuming
rho=0). Previous persistence attempts failed because (a) no signal_artifact was
embedded and (b) one attempt used the artifact dict as a FILENAME (Errno 36).

This script:
  1. Re-evaluates all candidates in the STRICT gate namespace
     (eval(expr, {'__builtins__':{}}, {'pd':pd,'np':np,'close':panel}) on the
     UNION panel of the 15 tradable assets). Candidates whose expression cannot
     be evaluated there (e.g. uses VIX / beta()) are marked NON-RECOVERABLE and
     are NOT persisted (the gate would quarantine them again).
  2. Recomputes validation metrics (IC/ICIR h=10, hit, decay, coverage,
     turnover, regime splits).
  3. Computes pairwise pooled |rho| among gate-recoverable passers and admits
     greedily by quality = |IC|*|ICIR| subject to max |rho| < 0.5 vs admitted.
  4. Writes factors/<factor_id>.json with full schema + embedded signal_artifact
     (row-major daily panel, NaN->null, rounded to 6dp).
  5. Reads back every file and verifies JSON validity / id / status / gates /
     artifact dims.
  6. Refreshes factors/factor_ensemble.json with the new admission set.
"""
import sys, json, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import (build_panel, forward_returns, spearman_ic,
                        mean_rank_turnover, ADMISSION_HORIZON, HORIZONS,
                        MIN_ASSETS, VISIBLE)

prices = build_panel()
panel = pd.DataFrame(prices)
env = {"pd": pd, "np": np, "close": panel}

RET = "close.pct_change()"
CANDIDATES = {
    # ---- recoverable ensemble factors (re-persist with artifact) ----
    "mom_10d_skip5":   "close.shift(5) / close.shift(15) - 1.0",
    "mom_120d_skip5":  "close.shift(5) / close.shift(125) - 1.0",
    "vol_of_vol20x60": "close.pct_change().rolling(20, min_periods=5).std().rolling(60, min_periods=15).std()",
    # ---- cycle-19 passers ----
    "gain_loss_20":    f"{RET}.clip(lower=0).rolling(20, min_periods=10).mean() / ({RET}.clip(upper=0).rolling(20, min_periods=10).mean().abs() + 1e-9)",
    "mom30_vol60":     "(close.shift(5)/close.shift(35)-1.0) / close.pct_change().rolling(60, min_periods=15).std()",
    "mom10_vol20":     "(close.shift(5)/close.shift(15)-1.0) / close.pct_change().rolling(20, min_periods=5).std()",
    "mom60_vol20":     "(close.shift(5)/close.shift(65)-1.0) / close.pct_change().rolling(20, min_periods=5).std()",
    "zscore_252":      "(close - close.rolling(252, min_periods=30).mean()) / close.rolling(252, min_periods=30).std()",
    # ---- explicitly non-recoverable (kept for the record, never persisted) ----
    "vix_beta_cond_60x20": "-beta(asset_ret, VIX_ret, 60) * (VIX/VIX.shift(20) - 1.0)",
}

META = {
    "mom_10d_skip5":   ("Short Momentum 10d (skip 5d)", "10-day price momentum with 5-day skip (return from t-15 to t-5) to avoid short-term reversal.", ["close"], {"lookback": 10, "skip": 5}, ["momentum", "cross-asset"]),
    "mom_120d_skip5":  ("Momentum 120d (skip 5d)", "120-day price momentum with 5-day skip (return from t-125 to t-5); medium-horizon trend with reversal skip.", ["close"], {"lookback": 120, "skip": 5}, ["momentum", "trend"]),
    "vol_of_vol20x60": ("Volatility of volatility 20x60", "60-day std of 20-day realized volatility; high = unstable/regime-shifting vol, low = calm vol regime.", ["close"], {"short_win": 20, "long_win": 60, "min_periods_short": 5, "min_periods_long": 15}, ["volatility", "regime"]),
    "gain_loss_20":    ("20d Gain/Loss ratio", "Ratio of mean positive daily return to mean |negative| daily return over 20d; skew/trend-quality of the return path.", ["close"], {"window": 20, "min_periods": 10}, ["trend", "quality", "skew"]),
    "mom30_vol60":     ("Risk-adjusted momentum 30d/vol60", "30-day momentum (skip 5) scaled by 60-day realized vol; stable trend per unit risk.", ["close"], {"lookback": 30, "skip": 5, "vol_win": 60, "min_periods": 15}, ["momentum", "risk-adjusted"]),
    "mom10_vol20":     ("Risk-adjusted momentum 10d/vol20", "10-day momentum (skip 5) scaled by 20-day realized vol; short trend per unit risk.", ["close"], {"lookback": 10, "skip": 5, "vol_win": 20, "min_periods": 5}, ["momentum", "risk-adjusted"]),
    "mom60_vol20":     ("Risk-adjusted momentum 60d/vol20", "60-day momentum (skip 5) scaled by 20-day realized vol; medium trend per unit short-term vol.", ["close"], {"lookback": 60, "skip": 5, "vol_win": 20, "min_periods": 5}, ["momentum", "risk-adjusted"]),
    "zscore_252":      ("252d Price z-score", "Z-score of close vs trailing 252d mean/std; how far the price level is from its 1y norm (carry/level regime).", ["close"], {"window": 252, "min_periods": 30}, ["level", "regime", "mean-reversion"]),
}

fwd10 = forward_returns(prices, ADMISSION_HORIZON)
signals, stats, recoverable = {}, {}, {}
print("=== strict-namespace eval (gate view: close+pd+np only) ===")
for fid, exp in CANDIDATES.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape
        if ok:
            signals[fid] = sig
            recoverable[fid] = True
        print(f"  {fid:18s} eval={'OK' if ok else 'BAD_SHAPE'}")
    except Exception as e:
        recoverable[fid] = False
        print(f"  {fid:18s} eval=FAIL({type(e).__name__}) NON-RECOVERABLE: {str(e)[:70]}")

print("\n=== validation metrics (admission h=10, gate-view signals) ===")
for fid, sig in signals.items():
    ics = spearman_ic(sig, fwd10)
    if len(ics) == 0:
        stats[fid] = dict(ic=0.0, icir=0.0, hit=0.0, n=0, cov=0.0, dates_ge8=0.0,
                          turn=np.nan, decay={}, regime={}, gate=False, quality=0.0)
        print(f"  {fid:18s} NO IC DATES")
        continue
    ic = float(ics.mean())
    icir = float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0
    hit = float((ics > 0).mean()) if ic >= 0 else float((ics < 0).mean())
    decay = {str(h): round(float(spearman_ic(sig, forward_returns(prices, h)).mean()), 4)
             for h in HORIZONS}
    cov = float(sig.notna().sum().sum()) / sig.size
    n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    turn = mean_rank_turnover(sig)
    gate = abs(ic) >= 0.007 and abs(icir) >= 0.084
    regime = {}
    for b0, b1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")]:
        sub = ics[(ics.index >= b0) & (ics.index <= b1)]
        if len(sub) >= 30:
            regime[f"{b0[:4]}-{b1[:4]}"] = {"ic": round(float(sub.mean()), 4),
                                            "icir": round(float(sub.mean() / sub.std()), 4) if sub.std() > 0 else 0.0,
                                            "n_dates": int(len(sub))}
    stats[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ics), cov=cov,
                      dates_ge8=n_ge8 / len(sig), turn=turn, decay=decay,
                      regime=regime, gate=gate, quality=abs(ic) * abs(icir))
    print(f"  {fid:18s} n={len(ics):5d} ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} "
          f"cov={cov:.3f} dates_ge8={n_ge8/len(sig):.3f} turn={turn:.3f} "
          f"gate={'PASS' if gate else 'no '} quality={stats[fid]['quality']:.5f}")

print("\n=== pairwise |rho| among gate-recoverable candidates (pooled) ===")
names = [fid for fid in signals]
rho = pd.DataFrame(index=names, columns=names, dtype=float)
for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i:
            continue
        both = pd.concat([signals[a].stack().rename("x"), signals[b].stack().rename("y")], axis=1).dropna()
        r = float(both["x"].corr(both["y"])) if len(both) > 100 else np.nan
        rho.loc[a, b] = r
        rho.loc[b, a] = r
for i, a in enumerate(names):
    row = "".join(f"{abs(rho.loc[a,b]):>7.3f}" if j < i else "       " for j, b in enumerate(names))
    print(f"  {a:18s}{row}")

print("\n=== greedy admission (gate passers, desc quality, max |rho| < 0.5 vs admitted) ===")
passers = [fid for fid in names if stats[fid]["gate"]]
order = sorted(passers, key=lambda f: -stats[f]["quality"])
admitted = []
for a in order:
    mx = max((abs(rho.loc[a, b]) for b in admitted), default=0.0)
    ok = mx < 0.5
    print(f"  {a:18s} quality={stats[a]['quality']:.5f} max_rho_vs_admitted={mx:.3f} -> {'ADMIT' if ok else 'reject'}")
    if ok:
        admitted.append(a)

print("\n=== final admission set ===")
for a in admitted:
    others = [b for b in admitted if b != a]
    mx = max((abs(rho.loc[a, b]) for b in others), default=0.0)
    print(f"  {a:18s} ic={stats[a]['ic']:+.4f} icir={stats[a]['icir']:+.4f} max_rho_vs_admitted={mx:.3f}")

# ---- embed signal artifact (row-major panel, NaN -> null) ----
dates = [d.strftime("%Y-%m-%d") for d in panel.index]
cols = list(panel.columns)
def to_artifact(sig):
    vals = []
    for d in panel.index:
        row = sig.loc[d]
        vals.append([None if pd.isna(v) else round(float(v), 6) for v in row])
    return {"format": "daily_panel", "dates": dates, "columns": cols, "values": vals}

CONTRACT = {"ic_threshold": 0.007, "icir_threshold": 0.084, "correlation_threshold": 0.5,
            "library_capacity": 30, "active_top_k": 10}
VALIDATED = f"2020-01-01..{VISIBLE}"
NOW = "2026-07-30"

print("\n=== writing factors/<factor_id>.json (with signal_artifact) ===")
written = []
for fid in admitted:
    sig = signals[fid]
    s = stats[fid]
    name, desc, deps, params, tags = META[fid]
    others = [b for b in admitted if b != fid]
    mx = max((abs(rho.loc[fid, b]) for b in others), default=0.0)
    payload = {
        "factor_id": fid,
        "factor_name": name,
        "version": "3.0.0",
        "calculation": {"expression": CANDIDATES[fid], "description": desc},
        "dependencies": deps,
        "parameters": params,
        "expected_direction": 1 if s["ic"] >= 0 else -1,
        "validation": {
            "status": "EFFECTIVE",
            "period": VALIDATED,
            "last_validated": NOW,
            "admission_horizon": ADMISSION_HORIZON,
            "regime_notes": "Validated on 15-instrument tradable cross-asset universe "
                            "(equity indices, commodities, crypto, yields), "
                            "h=10 daily Spearman rank IC, >=8 valid assets/date. Regime split: "
                            + "; ".join(f"{k}: ic={v['ic']:.3f} icir={v['icir']:.3f} n={v['n_dates']}"
                                        for k, v in s["regime"].items()),
            "metrics": {
                "ic": round(s["ic"], 4),
                "icir": round(s["icir"], 4),
                "ic_hit_ratio": round(s["hit"], 3),
                "n_ic_dates": s["n"],
                "coverage_asset_days": round(s["cov"], 4),
                "coverage_dates_ge8": round(s["dates_ge8"], 4),
                "turnover_10d_rank": round(s["turn"], 4) if np.isfinite(s["turn"]) else None,
                "decay_ic_by_horizon": {k: round(v, 4) for k, v in s["decay"].items()},
                "max_abs_library_correlation": round(mx, 4),
            },
        },
        "tags": tags,
        "benchmark_admission": {
            "contract": CONTRACT,
            "selected_metrics": {
                "ic": round(s["ic"], 4),
                "icir": round(s["icir"], 4),
                "metric_path": "validation.metrics",
                "max_abs_library_correlation": round(mx, 4),
                "correlation_path": "validation.metrics.max_abs_library_correlation",
                "quality": round(s["quality"], 8),
            },
            "admitted_at": f"{NOW}T00:00:00",
        },
        "signal_artifact": to_artifact(sig),
    }
    path = f"factors/{fid}.json"
    with open(path, "w") as f:
        json.dump(payload, f)
    written.append(fid)
    print(f"  wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)")

print("\n=== read-back verification ===")
ok_all = True
for fid in written:
    with open(f"factors/{fid}.json") as f:
        d = json.load(f)
    art = d.get("signal_artifact", {})
    metrics = d["validation"]["metrics"]
    checks = {
        "valid_json": True,
        "factor_id": d.get("factor_id") == fid,
        "status": d.get("validation", {}).get("status") == "EFFECTIVE",
        "ic_gate": abs(metrics["ic"]) >= 0.007,
        "icir_gate": abs(metrics["icir"]) >= 0.084,
        "artifact": art.get("format") == "daily_panel"
                    and len(art.get("dates", [])) == len(dates)
                    and len(art.get("values", [])) == len(dates)
                    and len(art.get("columns", [])) == 15
                    and all(len(r) == 15 for r in art.get("values", [])),
        "artifact_nonempty": sum(1 for row in art.get("values", []) for v in row if v is not None) > 100,
    }
    print(f"  {fid:18s} " + " ".join(f"{k}={v}" for k, v in checks.items()))
    ok_all &= all(checks.values())
print("ALL_VERIFIED" if ok_all else "VERIFY_FAILED")
if not ok_all:
    sys.exit(1)

# ---- refresh ensemble ----
total_q = sum(stats[f]["quality"] for f in admitted)
ens = {"schema_version": 1,
       "method": "quality_ic_tilt",
       "selected_factors": [{"factor_id": f, "weight": round(stats[f]["quality"] / total_q, 4),
                             "direction": 1 if stats[f]["ic"] >= 0 else -1} for f in admitted]}
with open("factors/factor_ensemble.json", "w") as f:
    json.dump(ens, f, indent=1)
print("\n=== factor_ensemble.json refreshed ===")
print(json.dumps(ens, indent=1))

json.dump({f: {k: (v if not isinstance(v, dict) else v) for k, v in stats[f].items()}
           for f in admitted}, open("scripts/_cycle19_persisted.json", "w"), indent=1, default=float)
print("\nsaved scripts/_cycle19_persisted.json")
