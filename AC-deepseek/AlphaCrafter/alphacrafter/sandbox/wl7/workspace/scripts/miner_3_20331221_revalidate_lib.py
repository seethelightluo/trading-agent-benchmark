"""miner_3 library revalidation through 2033-12-20."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2033-12-20"
cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)
fwd = ms.forward_ret(close, 10)
lib_panels = ms.library_panel(close, macro)
print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")


def max_lib_corr_excl_self(name, panels):
    cand = panels[name]
    flat = cand.stack(); best = 0.0; pairs = {}
    for n, p in panels.items():
        if n == name:
            continue
        pflat = p.reindex(cand.index).stack()
        df = pd.concat([flat.rename("f"), pflat.rename("p")], axis=1).dropna()
        if len(df) < 30:
            continue
        rho = float(df["f"].corr(df["p"])); pairs[n] = round(rho, 4)
        if abs(rho) > best:
            best = abs(rho)
    return best, pairs


factors = list(ms.ACTIVE_LIB.keys())
out = {"end": END, "horizon": 10, "results": {}}
print(f"{'factor':22s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s} {'maxrho':>7s}  GATE")
for name in factors:
    panel = lib_panels[name]
    st = ms.ic_stats(ms.daily_ic(panel, fwd), 10)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    mrho, pairs = max_lib_corr_excl_self(name, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE) else "fail"
    print(f"{name:<22s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    ic = ms.daily_ic(panel, fwd).dropna()
    row = {"ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
           "coverage_dates_ge8": cov["coverage_dates_ge8"], "turnover_10d": turn,
           "max_abs_library_correlation": round(mrho, 4), "library_pairs": pairs,
           "gate": gate}
    for lab, nd in (("1y", 365), ("6m", 183), ("3m", 91)):
        s = ic[ic.index >= ic.index.max() - np.timedelta64(nd, "D")]
        if len(s) == 0:
            continue
        m = s.mean(); sd = s.std(ddof=1); ir = m / sd if sd > 0 else np.nan
        print(f"    {lab}: IC {m:+.4f} ICIR {ir:+.3f} hit {(s>0).mean():.2f} n {len(s)}")
        row[f"recent_{lab}"] = {"ic": float(m), "icir": float(ir),
                                "hit": float((s > 0).mean()), "n": int(len(s))}
    out["results"][name] = row

with open("scripts/miner3_20331221_revalidation.json", "w") as h:
    json.dump(out, h, indent=2)
print("\nWrote scripts/miner3_20331221_revalidation.json")