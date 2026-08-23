"""miner_2 active-library re-validation visible through 2034-03-15."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover)

END = "2034-03-15"
close = load_close(END); macro = load_macro(END)
lib = library_panel(close, macro)
fwd = forward_ret(close, 10)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

factors = list(lib.keys())
out = {"end": END, "horizon": 10, "results": {}}
hdr = f"{'factor':24s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s} {'maxrho':>7s}  GATE"
print(hdr)
for name in factors:
    panel = lib[name]
    ic = daily_ic(panel, fwd)
    st = ic_stats(ic, 10)
    cov = coverage_stats(panel, fwd)
    turn = rank_turnover(panel, window=10)
    mrho, pairs = max_lib_corr(panel, lib)
    gate = "PASS" if (abs(st["ic"]) >= 0.0070 and abs(st["icir"]) >= 0.0840) else "fail"
    print(f"{name:24s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    row = {"ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
           "coverage_dates_ge8": cov["coverage_dates_ge8"], "turnover_10d": turn,
           "max_abs_library_correlation": round(mrho, 4),
           "library_pairs": {k: round(v, 4) for k, v in pairs.items()}}
    for lab, nd in (("1y", 365), ("6m", 183), ("3m", 91)):
        s = ic.dropna()
        s = s[s.index >= s.index.max() - np.timedelta64(nd, "D")]
        if len(s) == 0:
            continue
        m = s.mean(); sd = s.std(ddof=1); ir = m / sd if sd > 0 else np.nan
        print(f"     {lab}: IC {m:+.4f} ICIR {ir:+.3f} hit {(s>0).mean():.2f} n {len(s)}")
        row[f"recent_{lab}"] = {"ic": float(m), "icir": float(ir),
                                "hit": float((s > 0).mean()), "n": int(len(s))}
    out["results"][name] = row

with open("scripts/miner_2_20340315_revalidation.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nWrote scripts/miner_2_20340315_revalidation.json")