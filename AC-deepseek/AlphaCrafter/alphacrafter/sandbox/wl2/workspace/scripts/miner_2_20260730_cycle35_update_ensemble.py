"""miner_2 cycle35: update factor ensemble with newly admitted factors.

Currently admitted: mom20_volproxy60, usdjpy_beta_cond_120x60,
dxy_beta_cond_60x20, calmness_20 (4 factors).
Newly admitted this cycle (all have real .signal.npy artifacts):
  max_consec_gain_20 : IC=+0.0682 ICIR=+0.2310 q=0.015752 maxlibcorr=0.3418
  max_consec_loss_20 : IC=-0.0420 ICIR=-0.1432 q=0.006014 maxlibcorr=0.3253
  days_since_high_60 : IC=-0.0367 ICIR=-0.1185 q=0.004349 maxlibcorr=0.4281
Recompute q = |IC|*|ICIR|, weights = q/sum(q), direction = sign(IC).
Pairwise raw signal-artifact |rho| must remain < 0.5 gate; max checked after.
"""
import json
import sys
import numpy as np
from pathlib import Path
from scipy.stats import rankdata

sys.path.insert(0, "scripts")

ACTIVE = ["mom20_volproxy60", "usdjpy_beta_cond_120x60", "dxy_beta_cond_60x20",
          "calmness_20", "max_consec_gain_20", "max_consec_loss_20",
          "days_since_high_60"]

def artifact_path(d, m):
    """Resolve signal artifact path: newer factors store it in metrics,
    older ones at top level as a bare filename relative to factors/."""
    p = m.get("signal_artifact") or d.get("signal_artifact")
    assert p, f"no signal_artifact for {d['factor_id']}"
    if not p.startswith("factors/"):
        p = f"factors/{p}"
    return p

meta = {}
for fid in ACTIVE:
    d = json.load(open(f"factors/{fid}.json"))
    v = d["validation"]
    m = v["metrics"]
    art = artifact_path(d, m)
    assert Path(art).exists(), f"missing artifact for {fid}: {art}"
    meta[fid] = {
        "ic": m["ic"],
        "icir": m["icir"],
        "artifact": art,
        "last_validated": d.get("last_validated") or v.get("last_validated") or v.get("validated_at", "2026-07-30"),
        "category": d.get("category") or d.get("tags", ["unknown"])[0],
    }
    print(f"{fid:22s} ic={m['ic']:+.4f} icir={m['icir']:+.4f} "
          f"maxlibcorr={m.get('max_abs_library_correlation', float('nan')):.4f} "
          f"artifact={art}")

rows = []
for fid, mm in meta.items():
    q = abs(mm["ic"]) * abs(mm["icir"])
    rows.append({"factor_id": fid, "ic": mm["ic"], "icir": mm["icir"],
                 "q": q, "direction": 1 if mm["ic"] > 0 else -1})
rows.sort(key=lambda r: -r["q"])
total_q = sum(r["q"] for r in rows)
for r in rows:
    r["weight"] = r["q"] / total_q
    print(f"{r['factor_id']:22s} ic={r['ic']:+.4f} icir={r['icir']:+.4f} "
          f"q={r['q']:.6f} w={r['weight']:.6f} dir={r['direction']}")
print("weights sum:", sum(r["weight"] for r in rows))

# ---- pairwise |rho| gate on raw signal artifacts ----
arrs = {fid: np.load(meta[fid]["artifact"]) for fid in ACTIVE}
maxrho, pair = 0.0, None
K = len(ACTIVE)
for i in range(K):
    for j in range(i + 1, K):
        a, b = arrs[ACTIVE[i]], arrs[ACTIVE[j]]
        m = ~(np.isnan(a) | np.isnan(b))
        ra, rb = rankdata(a[m]), rankdata(b[m])
        rho = np.corrcoef(ra, rb)[0, 1]
        print(f"rho {ACTIVE[i]:22s} vs {ACTIVE[j]:22s} = {rho:+.4f} (n={m.sum()})")
        if abs(rho) > maxrho:
            maxrho, pair = abs(rho), (ACTIVE[i], ACTIVE[j], rho)
print(f"max pairwise |rho| = {maxrho:.4f} ({pair[0]} vs {pair[1]}) -> gate<0.5: {maxrho < 0.5}")
assert maxrho < 0.5, "correlation gate FAILED"

# ---- build ensemble ----
ens = json.load(open("factors/factor_ensemble.json"))
selected = []
for r in rows:
    fid = r["factor_id"]
    cat = {
        "mom20_volproxy60": "Momentum (vol-damped)",
        "usdjpy_beta_cond_120x60": "Macro-Conditional (JPY carry-risk regime)",
        "dxy_beta_cond_60x20": "Macro-Conditional (DXY beta)",
        "calmness_20": "Low-volatility (calmness)",
        "max_consec_gain_20": "Return-consistency (win streak)",
        "max_consec_loss_20": "Return-consistency (losing streak)",
        "days_since_high_60": "Recovery / trend-freshness",
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
    f"ACTIVE LIBRARY = {len(selected)} FACTORS with real .signal.npy artifacts.",
    "Newly admitted cycle35: max_consec_gain_20 (IC 0.0682/ICIR 0.2310, q 0.015752, max lib corr 0.3418), "
    "max_consec_loss_20 (IC -0.0420/ICIR -0.1432, q 0.006014, max lib corr 0.3253), "
    "days_since_high_60 (IC -0.0367/ICIR -0.1185, q 0.004349, max lib corr 0.4281).",
    "pos_freq_20 (IC 0.045/ICIR 0.147) NOT persisted: pairwise rank rho vs max_consec_gain_20 = +0.636 > 0.5 -> guaranteed quarantine.",
    "Pairwise raw signal-artifact rho gate re-checked for all 7 active factors: max = "
    f"{maxrho:.4f} ({pair[0]} vs {pair[1]}) < 0.5 gate.",
    "Concentration reduced from 4 to 7 factors; mom20_volproxy60 weight drops further "
    f"to {selected[0]['weight'] * 100:.1f}%.",
    "downside_dev_60 (DEPRECATED) NOT in active ensemble; signal artifact kept for correlation gate only.",
]
json.dump(ens, open("factors/factor_ensemble.json", "w"), indent=1, default=str)
print("\nwrote factors/factor_ensemble.json | n_selected =", len(selected),
      "| weights_sum =", ens["weights_sum"])
