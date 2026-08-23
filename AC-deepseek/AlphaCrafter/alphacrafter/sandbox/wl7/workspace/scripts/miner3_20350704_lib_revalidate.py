"""miner_3 library re-validation as of visible_through 2035-07-03.

Re-runs all 8 active library factors on the full panel through the current
visible window and reports per-horizon IC/ICIR/hit/coverage. Flags factors
failing the |IC|>=0.0070 or |ICIR|>=0.0840 gate (or with ICIR turning
significantly negative) for deprecation.
"""
import json
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2035-07-03"
cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)
print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")

lib = ms.library_panel(close, macro)
summary = {}
for name, panel in lib.items():
    res = ms.summarize(panel, close)
    cov_fwd = ms.forward_ret(close, 10)
    cov = ms.coverage_stats(panel, cov_fwd)
    h = 10
    r = res[h]
    gate = abs(r["ic"]) >= ms.IC_GATE and abs(r["icir"]) >= ms.ICIR_GATE
    summary[name] = {
        "ic10": r["ic"], "icir10": r["icir"], "hit10": r["hit"], "n10": r["n"],
        "decay": {str(k): round(v["ic"], 4) for k, v in res.items()},
        "coverage": cov, "gate_pass": gate,
    }
    print(f"{name:28s} IC10={r['ic']:+.5f} ICIR10={r['icir']:+.5f} "
          f"hit={r['hit']:.3f} n={r['n']} gate={'PASS' if gate else 'FAIL'}")

with open("scripts/miner3_20350704_lib_revalidate.json", "w") as f:
    json.dump(summary, f, indent=2, default=str)
print("saved scripts/miner3_20350704_lib_revalidate.json")