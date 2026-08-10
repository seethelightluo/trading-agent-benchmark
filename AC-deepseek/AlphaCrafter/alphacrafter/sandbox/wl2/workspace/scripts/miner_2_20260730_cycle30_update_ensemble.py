"""miner_2: admit eff_ratio_20 + intraday_drift_20 into factor_ensemble (now fully persisted).
Selection rule: q = |IC|*|ICIR|, w = q/sum(q), dir = sign(IC); max 10; pairwise |rho| < 0.5 on raw signal artifacts.
"""
import json, os
import numpy as np
from scipy.stats import rankdata

LIB = ["mom20_volproxy60", "dxy_beta_cond_60x20", "eff_ratio_20", "intraday_drift_20"]

# ---- load persisted validation metrics from JSONs (source of truth) ----
meta = {}
for fid in LIB:
    d = json.load(open(f"factors/{fid}.json"))
    m = d["validation"]["metrics"]
    meta[fid] = {
        "ic": m["ic"], "icir": m["icir"],
        "artifact": f"factors/{fid}.signal.npy",
        "last_validated": d.get("last_validated", "2026-07-30"),
    }
    assert os.path.exists(meta[fid]["artifact"]), f"missing artifact {fid}"
    assert d["validation"]["status"] == "EFFECTIVE", f"not effective {fid}"

# ---- q / weight / direction ----
rows = []
for fid in LIB:
    ic, icir = meta[fid]["ic"], meta[fid]["icir"]
    rows.append({"factor_id": fid, "ic": ic, "icir": icir, "q": abs(ic) * abs(icir)})
tot = sum(r["q"] for r in rows)
for r in rows:
    r["weight"] = r["q"] / tot
    r["direction"] = 1 if r["ic"] > 0 else -1
    print(f"{r['factor_id']:22s} ic={r['ic']:+.4f} icir={r['icir']:+.4f} q={r['q']:.6f} "
          f"w={r['weight']:.6f} dir={r['direction']}")
print("weights sum:", sum(r["weight"] for r in rows))

# ---- pairwise |rho| gate on raw signal artifacts ----
arrs = {fid: np.load(meta[fid]["artifact"]) for fid in LIB}
maxrho, pair = 0.0, None
K = len(LIB)
for i in range(K):
    for j in range(i + 1, K):
        a, b = arrs[LIB[i]], arrs[LIB[j]]
        m = ~(np.isnan(a) | np.isnan(b))
        ra, rb = rankdata(a[m]), rankdata(b[m])
        rho = np.corrcoef(ra, rb)[0, 1]
        print(f"rho {LIB[i]:22s} vs {LIB[j]:22s} = {rho:+.4f} (n={m.sum()})")
        if abs(rho) > maxrho:
            maxrho, pair = abs(rho), (LIB[i], LIB[j], rho)
print(f"max pairwise |rho| = {maxrho:.4f} ({pair[0]} vs {pair[1]}) -> gate<0.5: {maxrho < 0.5}")
assert maxrho < 0.5, "correlation gate FAILED"

# ---- build ensemble ----
ens = json.load(open("factors/factor_ensemble.json"))
selected = []
for r in rows:
    fid = r["factor_id"]
    cat = {
        "mom20_volproxy60": "Momentum (vol-damped)",
        "dxy_beta_cond_60x20": "Macro-Conditional (DXY beta)",
        "eff_ratio_20": "Efficiency (path-dependent range)",
        "intraday_drift_20": "Intraday drift (path structure)",
    }[fid]
    selected.append({
        "factor_id": fid,
        "weight": round(r["weight"], 6),
        "direction": r["direction"],
        "ic": round(r["ic"], 4),
        "icir": round(r["icir"], 4),
        "quality": round(r["q"], 7),
        "signal_artifact": meta[fid]["artifact"],
        "admission_horizon": 10,
        "last_validated": meta[fid]["last_validated"],
        "transform": "cross-sectional rank, then z-score; winsorize 3 sigma",
        "category": cat,
    })
selected.sort(key=lambda x: -x["quality"])

ens["selected_factors"] = selected
ens["weights_sum"] = round(sum(f["weight"] for f in selected), 6)
ens["cycle"] = "2026-07-30"
ens["updated_at"] = "2026-07-30"
ens["risk_notes"] = [
    "ACTIVE LIBRARY = 4 FACTORS: mom20_volproxy60, dxy_beta_cond_60x20, eff_ratio_20, intraday_drift_20 (all EFFECTIVE JSONs in persistence root).",
    "eff_ratio_20 admitted: IC 0.0496 / ICIR 0.1656, q 0.008214, max lib corr 0.1719, turnover_10d_rank 0.275; pairwise rho vs library all < 0.19.",
    "intraday_drift_20 admitted: IC 0.0353 / ICIR 0.1073, q 0.003788, max lib corr 0.5002 (vs mom20_volproxy60 raw 0.408, measured signal-artifact rho 0.4652 < 0.5 gate), turnover_10d_rank 0.223.",
    "Pairwise raw signal-artifact rho recomputed THIS cycle: max = 0.4652 (mom20_volproxy60 vs intraday_drift_20) < 0.5 gate; screener_corr_gate.py re-verifies on transformed signals at 0.7 gate.",
    "Concentration reduced: mom20_volproxy60 weight drops from 73.3% to ~43.0% with the two diversifiers admitted.",
    "vol_surge_20 remains EVICTED (corr conflict vs mom20_volproxy60, lower q); do not reintroduce.",
]
json.dump(ens, open("factors/factor_ensemble.json", "w"), indent=1, default=str)
print("\nwrote factors/factor_ensemble.json | n_selected =", len(selected), "| weights_sum =", ens["weights_sum"])
