"""miner_1 library revalidation through 2034-03-14 (current date 2034-03-15)."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2034-03-14"
cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)
fwd = ms.forward_ret(close, 10)
lib = ms.library_panel(close, macro)
print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")

out = {"end": END, "horizon": 10, "results": {}}
print(f"{'factor':24s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s}  GATE")
for name in ms.ACTIVE_LIB:
    panel = lib[name]
    st = ms.ic_stats(ms.daily_ic(panel, fwd), 10)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    mrho, pairs = ms.max_lib_corr(name, lib)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE) else "fail"
    print(f"{name:<22s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f}  {gate}")
    ic = ms.daily_ic(panel, fwd).dropna()
    row = {"ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
           "coverage_dates_ge8": cov["coverage_dates_ge8"], "turnover_10d": turn,
           "max_abs_library_correlation": round(mrho, 4), "gate": gate}
    for lab, nd in (("6m", 183), ("3m", 91)):
        s = ic[ic.index >= ic.index.max() - np.timedelta64(nd, "D")]
        if len(s) == 0:
            continue
        m = s.mean(); sd = s.std(ddof=1); ir = m / sd if sd > 0 else float("nan")
        print(f"    {lab}: IC {m:+.4f} ICIR {ir:+.3f} hit {(s>0).mean():.2f} n {len(s)}")
        row[f"recent_{lab}"] = {"ic": float(m), "icir": float(ir), "hit": float((s>0).mean()), "n": int(len(s))}
    out["results"][name] = row

with open("scripts/miner1_20340315_lib_reval.json", "w") as h:
    json.dump(out, h, indent=2)
print("\nWrote scripts/miner1_20340315_lib_reval.json")