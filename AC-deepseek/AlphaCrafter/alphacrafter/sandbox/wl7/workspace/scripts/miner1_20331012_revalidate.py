"""miner_1 revalidation of active library at visible_through 2033-10-11."""
import sys, json
sys.path.insert(0, "scripts")
from miner_shared import (
    load_close, load_macro, forward_ret, daily_ic, ic_stats, summarize,
    rank_turnover, coverage_stats, library_panel, max_lib_corr,
    IC_GATE, ICIR_GATE,
)
import pandas as pd
import numpy as np

END = "2033-10-11"
close = load_close(END)
macro = load_macro(END)
lib = library_panel(close, macro)

print("=== Active library revalidation @ END=%s ===" % END)
print("dates:", close.shape[0], "assets:", close.shape[1])

res = []
for name, panel in lib.items():
    ic = daily_ic(panel, forward_ret(close, 1))
    st = ic_stats(ic, 1)
    full = summarize(panel, close)
    turn = rank_turnover(panel)
    cov = coverage_stats(panel, forward_ret(close, 1))
    gate = (abs(st["ic"]) >= IC_GATE) and (abs(st["icir"]) >= ICIR_GATE)
    best, pairs = max_lib_corr(panel, {k: v for k, v in lib.items() if k != name})
    # recent sub-windows on the 10d forward-return IC
    ic10 = daily_ic(panel, forward_ret(close, 10)).dropna()
    row = dict(
        name=name, ic=round(st["ic"], 6), icir=round(st["icir"], 6),
        hit=round(st["hit"], 4), n=st["n"],
        ic_h5=round(full[5]["ic"], 6), icir_h5=round(full[5]["icir"], 6),
        ic_h10=round(full[10]["ic"], 6), icir_h10=round(full[10]["icir"], 6),
        turn=round(turn, 4),
        covAD=round(cov["coverage_asset_days"], 4),
        covD8=round(cov["coverage_dates_ge8"], 4),
        maxrho=round(best, 4), pairs=pairs, gate=bool(gate),
    )
    for lab, nd in (("recent_1y", 365), ("recent_6m", 183), ("recent_3m", 91)):
        s = ic10[ic10.index >= ic10.index.max() - np.timedelta64(nd, "D")]
        if len(s) > 0:
            m = s.mean(); sd = s.std(ddof=1)
            ir = m / sd if sd > 0 else np.nan
            row[lab] = dict(ic=round(float(m), 4), icir=round(float(ir), 3),
                            hit=round(float((s > 0).mean()), 3), n=int(len(s)))
    res.append(row)
    print(f"{name:24s} IC(h1)={st['ic']:+.4f} ICIR(h1)={st['icir']:+.4f} hit={st['hit']:.3f} "
          f"n={st['n']} ic5={full[5]['ic']:+.4f} ic10={full[10]['ic']:+.4f} "
          f"ir10={full[10]['icir']:+.3f} turn={turn:.2f} gate={gate} maxrho={best:.3f}")
    if "recent_3m" in row:
        r = row["recent_3m"]
        print(f"      recent_3m10: IC {r['ic']:+.4f} ICIR {r['icir']:+.3f} hit {r['hit']:.2f} n {r['n']}")

json.dump(res, open("scripts/miner1_20331012_revalidation.json", "w"), indent=1)
print("saved: scripts/miner1_20331012_revalidation.json")