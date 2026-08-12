"""miner_2: persist 2028-06-09 revalidation results into factor library files.
Reval gate: |IC| >= 0.0070 and |ICIR| >= 0.0840 (1d horizon, full 2021+ window).
- PASS factors: update validation.status=EFFECTIVE, last_validated, append revalidation entry.
- FAIL factors (mine): mark DEPRECATED and rename file with _deprecated suffix.
Only touches miner2_* factor files. Shared factors (mom/vol_of_vol/vix_beta) left intact.
"""
import json
import os
import glob
import shutil
from datetime import datetime

REVAL = json.load(open("scripts/miner2_20280609_reval_results.json"))
NOW = "2028-06-09T00:00:00Z"

# map factor file -> short factor id used in reval results
FILE_ID = {
    "miner2_20260715_id_rev_1d": "id_rev_1d",
    "miner2_20260715_nbody_1d": "nbody_1d",
    "miner2_20260715_nclv_1d": "nclv_1d",
    "miner2_20260715_nclv_2d": "nclv_2d",
    "miner2_20260715_nclv_3d": "nclv_3d",
    "miner2_20260715_nclv_5d": "nclv_5d",
    "miner2_20260715_rev_1d": "rev_1d",
    "miner2_20260715_rev_1d_vs": "rev_1d_vs",
    "miner2_20260715_rev_2d": "rev_2d",
    "miner2_20260715_rev_3d": "rev_3d",
    "miner2_20260715_rev_5d": "rev_5d",
}

GATE_IC = 0.007
GATE_ICIR = 0.084

for fname, short in FILE_ID.items():
    path = f"factors/{fname}.json"
    if not os.path.exists(path):
        print("MISSING", path)
        continue
    d = json.load(open(path))
    r = REVAL.get(short)
    if r is None:
        print("no reval row for", short)
        continue
    ic, icir = r["ic"], r["icir"]
    passed = (abs(ic) >= GATE_IC) and (abs(icir) >= GATE_ICIR)
    status = "EFFECTIVE" if passed else "DEPRECATED"
    v = d.setdefault("validation", {})
    v["status"] = status
    v["last_validated"] = NOW
    v["timeliness"] = f"re-validated {NOW[:10]}; quarterly cadence"
    metrics = v.setdefault("metrics", {})
    metrics["ic1_reval"] = round(ic, 4)
    metrics["icir1_reval"] = round(icir, 4)
    metrics["hit1_reval"] = round(r["hit"], 4)
    metrics["n_dates_reval"] = r["n"]
    metrics["coverage_reval"] = round(r["coverage"], 4)
    metrics["turnover10_reval"] = round(r["turnover10"], 4)
    metrics["recent_ic1_400d"] = round(r["ric"], 4)
    metrics["recent_icir1_400d"] = round(r["ricir"], 4)
    metrics["max_abs_library_correlation_reval"] = round(r["maxlib"], 4)
    revals = v.setdefault("revalidations", [])
    revals.append({
        "date": NOW,
        "horizon": "1d forward, daily cross-sectional Spearman rank IC",
        "universe": "15 tradable cross-asset instruments",
        "min_valid_per_date": 8,
        "gates": {"abs_ic_min": GATE_IC, "abs_icir_min": GATE_ICIR},
        "metrics": {
            "ic_full": round(ic, 4),
            "icir_full": round(icir, 4),
            "hit_full": round(r["hit"], 4),
            "n_dates_full": r["n"],
            "ic_rec_400d": round(r["ric"], 4),
            "icir_rec_400d": round(r["ricir"], 4),
            "coverage": round(r["coverage"], 4),
            "turnover10": round(r["turnover10"], 4),
            "max_abs_library_correlation": round(r["maxlib"], 4),
            "gate": "PASS" if passed else "FAIL",
        },
        "status": status,
        "note": ("Quarterly re-validation 2028-06-09: passes |IC|>=0.007 and |ICIR|>=0.084 "
                 "on full sample." if passed else
                 "Quarterly re-validation 2028-06-09: FAILS gate (|IC|>=0.007 and |ICIR|>=0.084). "
                 "Marked DEPRECATED."),
    })
    v["revalidation_summary"] = {
        "latest_full_ic": round(ic, 4),
        "latest_full_icir": round(icir, 4),
        "latest_recent_ic": round(r["ric"], 4),
        "latest_recent_icir": round(r["ricir"], 4),
        "gate": "PASS" if passed else "FAIL",
    }
    with open(path, "w") as fh:
        json.dump(d, fh, indent=1, default=float)
    print(f"updated {fname}: status={status} ic={ic:.4f} icir={icir:.4f}")

# deprecate failing files by renaming with _deprecated suffix
for fname, short in FILE_ID.items():
    path = f"factors/{fname}.json"
    if not os.path.exists(path):
        continue
    d = json.load(open(path))
    if d.get("validation", {}).get("status") == "DEPRECATED":
        new_path = f"factors/{fname}_deprecated.json"
        shutil.move(path, new_path)
        print("DEPRECATED renamed:", path, "->", new_path)

print("\ndone")
