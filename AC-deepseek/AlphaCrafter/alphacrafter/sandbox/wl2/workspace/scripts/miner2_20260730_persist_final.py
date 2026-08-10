"""miner_2 2026-07-30: persist the final non-redundant admission set to factors/.

For each admitted factor:
  1. Re-evaluate the signal in the STRICT gate namespace (close panel + pd + np only).
  2. Recompute validation metrics (IC/ICIR h=10, hit, decay, coverage, turnover, regime).
  3. Compute max_abs_library_correlation vs the other admitted factors (real artifacts).
  4. Write factors/<factor_id>.json with full schema + embedded signal_artifact
     (row-major daily panel, NaN->null) so the post-Miner gate can recover the
     real signal without re-evaluation or namespace assumptions.
  5. Read back each file, verify JSON validity / ids / status / thresholds / artifact.
Then refresh factors/factor_ensemble.json with the new admission set.
"""
import sys, json, glob, os
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner3_lib import (build_panel, forward_returns, spearman_ic,
                        mean_rank_turnover, ADMISSION_HORIZON, HORIZONS,
                        MIN_ASSETS, VISIBLE)

prices = build_panel()
panel = pd.DataFrame(prices)          # gate namespace object
env = {"pd": pd, "np": np, "close": panel}

CANDIDATES = {
    "spx_corr60":      "close.pct_change().rolling(60, min_periods=15).corr(close['SPX'].pct_change())",
    "mom_20d_skip5":   "close.shift(5) / close.shift(25) - 1.0",
    "vol_of_vol20x60": "close.pct_change().rolling(20, min_periods=5).std().rolling(60, min_periods=15).std()",
    "mom_180d_skip5":  "close.shift(5) / close.shift(185) - 1.0",
    "range_pos_252":   "(close - close.rolling(252, min_periods=30).min()) / (close.rolling(252, min_periods=30).max() - close.rolling(252, min_periods=30).min())",
}
META = {
    "spx_corr60":      ("SPX 60d rolling correlation", "60-day rolling correlation of asset daily returns with SPX returns (cross-asset beta/regime factor). High values = high beta to US equities.", ["close"], {"window": 60, "min_periods": 15, "ref": "SPX"}, ["cross-asset", "beta", "regime"]),
    "mom_20d_skip5":   ("Momentum 20d (skip 5d)", "20-day price momentum ending 5 days ago (return from t-25 to t-5), skipping the most recent week to avoid short-term reversal.", ["close"], {"lookback": 20, "skip": 5}, ["momentum", "cross-asset"]),
    "vol_of_vol20x60": ("Volatility of volatility 20x60", "60-day std of 20-day realized volatility: high value = unstable/regime-shifting vol, low = calm vol regime.", ["close"], {"short_win": 20, "long_win": 60, "min_periods_short": 5, "min_periods_long": 15}, ["volatility", "regime"]),
    "mom_180d_skip5":  ("Momentum 180d (skip 5d)", "180-day price momentum ending 5 days ago (return from t-185 to t-5); long-horizon trend with short-term reversal skip.", ["close"], {"lookback": 180, "skip": 5}, ["momentum", "trend"]),
    "range_pos_252":   ("252d Range Position", "Position of close within its trailing 252-day high-low range (0=at low, 1=at high). Trend/mean-reversion regime location.", ["close"], {"window": 252, "min_periods": 30}, ["trend", "range", "regime"]),
}

fwd10 = forward_returns(prices, ADMISSION_HORIZON)
signals, stats = {}, {}
print("=== strict-namespace eval (close+pd+np) ===")
for fid, exp in CANDIDATES.items():
    try:
        sig = eval(exp, {"__builtins__": {}}, env)
        ok = isinstance(sig, pd.DataFrame) and sig.shape == panel.shape
        signals[fid] = sig
        print(f"  {fid:18s} eval={'OK' if ok else 'BAD_SHAPE'}")
    except Exception as e:
        print(f"  {fid:18s} eval=FAIL {type(e).__name__}: {str(e)[:90]}")

print("\n=== validation metrics (h=10 admission) ===")
for fid, sig in signals.items():
    ics = spearman_ic(sig, fwd10)
    ic = float(ics.mean())
    icir = float(ics.mean() / ics.std()) if ics.std() > 0 else 0.0
    hit = float((ics > 0).mean()) if ic >= 0 else float((ics < 0).mean())
    decay = {str(h): round(float(spearman_ic(sig, forward_returns(prices, h)).mean()), 4) for h in HORIZONS}
    cov = float(sig.notna().sum().sum()) / sig.size
    n_ge8 = sum(1 for d in sig.index if sig.loc[d].notna().sum() >= MIN_ASSETS)
    turn = mean_rank_turnover(sig)
    regime = {}
    for b0, b1 in [("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"),
                   ("2023-01-01", "2024-12-31"), ("2025-01-01", "2026-12-31")]:
        sub = ics[(ics.index >= b0) & (ics.index <= b1)]
        if len(sub) >= 30:
            regime[f"{b0[:4]}-{b1[:4]}"] = {"ic": round(float(sub.mean()), 4),
                                            "icir": round(float(sub.mean() / sub.std()), 4) if sub.std() > 0 else 0.0,
                                            "n_dates": int(len(sub))}
    stats[fid] = dict(ic=ic, icir=icir, hit=hit, n=len(ics), cov=cov, dates_ge8=n_ge8 / len(sig),
                      turn=turn, decay=decay, regime=regime)
    print(f"  {fid:18s} n={len(ics):5d} ic={ic:+.4f} icir={icir:+.4f} hit={hit:.3f} "
          f"cov={cov:.3f} dates_ge8={n_ge8/len(sig):.3f} turn={turn:.3f}")

print("\n=== max_abs_library_correlation (vs other admitted, real artifacts) ===")
for fid in signals:
    others = {b: s for b, s in signals.items() if b != fid}
    mx, names = 0.0, []
    for b, s in others.items():
        both = pd.concat([signals[fid].stack().rename("x"), s.stack().rename("y")], axis=1).dropna()
        if len(both) < 50:
            continue
        r = float(both["x"].corr(both["y"]))
        if abs(r) > mx:
            mx, names = abs(r), [b]
        elif abs(r) == mx:
            names.append(b)
    stats[fid]["maxrho"] = round(mx, 4)
    stats[fid]["maxrho_names"] = names
    print(f"  {fid:18s} max_rho_vs_admitted={mx:.4f} (vs {names})")

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

print("\n=== writing factors/<factor_id>.json ===")
for fid, sig in signals.items():
    name, desc, deps, params, tags = META[fid]
    s = stats[fid]
    payload = {
        "factor_id": fid,
        "factor_name": name,
        "version": "2.0.0",
        "calculation": {
            "expression": CANDIDATES[fid],
            "description": desc,
        },
        "dependencies": deps,
        "parameters": params,
        "expected_direction": 1,
        "validation": {
            "status": "EFFECTIVE",
            "period": VALIDATED,
            "last_validated": NOW,
            "admission_horizon": ADMISSION_HORIZON,
            "regime_notes": "Validated on 15-instrument tradable cross-asset universe "
                            "(equity indices, commodities, crypto, yields). Regime split: "
                            + "; ".join(f"{k}: ic={v['ic']:.3f} icir={v['icir']:.3f} n={v['n_dates']}"
                                        for k, v in s["regime"].items()),
            "metrics": {
                "ic": round(s["ic"], 4),
                "icir": round(s["icir"], 4),
                "ic_hit_ratio": round(s["hit"], 3),
                "n_ic_dates": s["n"],
                "coverage_asset_days": round(s["cov"], 4),
                "coverage_dates_ge8": round(s["dates_ge8"], 4),
                "turnover_10d_rank": round(s["turn"], 4),
                "decay_ic_by_horizon": {k: round(v, 4) for k, v in s["decay"].items()},
                "max_abs_library_correlation": s["maxrho"],
            },
        },
        "tags": tags,
        "benchmark_admission": {
            "contract": CONTRACT,
            "selected_metrics": {
                "ic": round(s["ic"], 4),
                "icir": round(s["icir"], 4),
                "metric_path": "validation.metrics",
                "max_abs_library_correlation": s["maxrho"],
                "correlation_path": "validation.metrics.max_abs_library_correlation",
                "quality": round(abs(s["ic"]) * abs(s["icir"]), 8),
            },
            "admitted_at": f"{NOW}T00:00:00",
        },
        "signal_artifact": to_artifact(sig),
    }
    path = f"factors/{fid}.json"
    with open(path, "w") as f:
        json.dump(payload, f)
    print(f"  wrote {path}  ({os.path.getsize(path)/1024:.0f} KB)")

# ---- read-back verification ----
print("\n=== read-back verification ===")
ok_all = True
for fid in signals:
    with open(f"factors/{fid}.json") as f:
        d = json.load(f)
    art = d.get("signal_artifact", {})
    checks = {
        "valid_json": True,
        "factor_id": d.get("factor_id") == fid,
        "status": d.get("validation", {}).get("status") == "EFFECTIVE",
        "ic_gate": abs(d["validation"]["metrics"]["ic"]) >= 0.007,
        "icir_gate": abs(d["validation"]["metrics"]["icir"]) >= 0.084,
        "artifact": art.get("format") == "daily_panel" and len(art.get("dates", [])) == len(dates)
                    and len(art.get("values", [])) == len(dates) and len(art.get("columns", [])) == 15,
        "expr_self_contained": all(tok not in d["calculation"]["expression"]
                                   for tok in ["std(", "pct_change.", "VIX", "DXY"]),
    }
    print(f"  {fid:18s} " + " ".join(f"{k}={v}" for k, v in checks.items()))
    ok_all &= all(checks.values())
print("ALL_VERIFIED" if ok_all else "VERIFY_FAILED")
