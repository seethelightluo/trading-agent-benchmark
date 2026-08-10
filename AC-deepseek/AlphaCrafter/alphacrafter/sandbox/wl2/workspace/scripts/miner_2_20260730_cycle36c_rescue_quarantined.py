"""miner_2 cycle36c: rescue + re-validate quarantined factors with real artifacts.

Four factors were quarantined by the deterministic gate with reason
"factor has no recoverable signal artifact" -- i.e., they passed IC/ICIR but had
no .signal.npy persisted. They can be re-admitted if, when recomputed NOW with
real artifacts, they still pass:
  - abs(IC)  >= 0.0070 @10d
  - abs(ICIR) >= 0.0840 @10d
  - max_abs_library_correlation < 0.5
  - pairwise |rho| vs all other candidates < 0.5 (checked after)

Candidates (exact definitions from their quarantined JSONs):
  - mom_10d_skip5   : close.shift(5)/close.shift(15) - 1
  - mom_120d_skip5  : close.shift(5)/close.shift(125) - 1
  - vol_of_vol20x60 : std(pct_change,20).rolling(60).std()
  - vix_beta_cond_60x20: -beta(asset_ret, VIX_ret, 60) * (VIX/VIX.shift(20)-1)

All validation metrics recomputed from scratch (no reuse of old claims).
"""
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, "scripts")
from miner2_lib import (TRADABLES, load_close_panel, macro_series, per_asset,
                        compute_ic, forward_returns, validate_factor,
                        regime_breakdown, report, save_signal_artifact,
                        panel_rank_corr)

cl = load_close_panel()
idx = cl.index
print(f"panel {cl.shape}, last date {cl.index[-1].date()}")

# ---- candidate signals ----
cands = {}
cands["mom_10d_skip5"] = per_asset(cl, lambda s: s.shift(5) / s.shift(15) - 1.0)
cands["mom_120d_skip5"] = per_asset(cl, lambda s: s.shift(5) / s.shift(125) - 1.0)
cands["vol_of_vol20x60"] = per_asset(
    cl, lambda s: s.pct_change().rolling(20).std().rolling(60).std())

vix = macro_series("VIX")
vix_ret = vix.pct_change()
vix_20 = vix / vix.shift(20) - 1.0
beta_parts = {}
for a in cl.columns:
    s = cl[a].dropna()
    ar = s.pct_change()
    df = pd.concat([ar.rename("a"), vix_ret.reindex(ar.index).rename("v")], axis=1).dropna()
    b = df["a"].rolling(60).cov(df["v"]) / df["v"].rolling(60).var()
    beta_parts[a] = b.reindex(idx)
beta_panel = pd.DataFrame(beta_parts, index=idx)
cands["vix_beta_cond_60x20"] = -beta_panel.mul(vix_20.reindex(idx), axis=0)

# ---- library: all real .signal.npy artifacts with matching shape ----
lib = {}
for f in sorted(Path("factors").glob("*.signal.npy")):
    arr = np.load(f)
    if arr.shape == cl.shape:
        fid = f.name.replace(".signal.npy", "")
        if fid != "downside_dev_60":
            lib[fid] = pd.DataFrame(arr, index=idx, columns=cl.columns)
print(f"[lib] loaded {len(lib)} artifacts: {sorted(lib.keys())}\n")

fwd = {str(h): forward_returns(cl, h) for h in (1, 2, 3, 5, 10, 20)}

print("=== VALIDATION (admission horizon=10d) ===")
results = {}
for name, f in cands.items():
    m = validate_factor(f, cl, library=lib, fwd_cache=fwd)
    p = report(name, m)
    print("    decay:", m["decay_ic_by_horizon"])
    strong = {k: v for k, v in m.get("library_pairwise_corr", {}).items() if abs(v) > 0.1}
    print("    pairwise(>0.1):", strong)
    print()
    results[name] = {"metrics": m, "pass": p}

print("=== REGIME BREAKDOWN (10d IC) ===")
for name, f in cands.items():
    ic_ser = compute_ic(f, fwd["10"]).dropna()
    reg = regime_breakdown(ic_ser)
    print(f"  {name:18s} | " + " | ".join(
        f"{k}: ic={v['ic']:+.4f} icir={v['icir']:+.3f} n={v['n_dates']}"
        for k, v in reg.items()))

print("\n=== PAIRWISE RANK |rho| AMONG CANDIDATES ===")
names = list(cands)
for i in range(len(names)):
    for j in range(i + 1, len(names)):
        rho = panel_rank_corr(cands[names[i]], cands[names[j]])
        print(f"  rho {names[i]:18s} vs {names[j]:18s} = {rho:+.4f}")

json.dump({k: {"metrics": v["metrics"], "pass": v["pass"]} for k, v in results.items()},
          open("scripts/miner_2_20260730_cycle36c_results.json", "w"), indent=1, default=str)
print("\nwrote scripts/miner_2_20260730_cycle36c_results.json")
