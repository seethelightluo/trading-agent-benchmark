"""miner_2 active library revalidation through 2032-04-30.
Read-only; runs validation only. No account/date writes.
"""
import json, sys
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import load_close, load_macro, forward_ret, daily_ic, ic_stats, library_panel, IC_GATE, ICIR_GATE

END = "2032-04-30"
close = load_close(END); macro = load_macro(END)
lib = library_panel(close, macro)
fwd = forward_ret(close, 10)
print(f"END={END} n_dates={len(close)} n_assets={close.shape[1]}")

rows = []
for name, f in lib.items():
    ic = daily_ic(f, fwd); st = ic_stats(ic, 10)
    s = ic.dropna()
    recent = s[s.index >= "2031-11-01"]
    r_ic = recent.mean() if len(recent) else np.nan
    r_icir = (recent.mean()/recent.std(ddof=1)) if len(recent)>2 and recent.std(ddof=1)>0 else np.nan
    gate = abs(st["ic"]) >= IC_GATE and abs(st["icir"]) >= ICIR_GATE
    # per-year h10 IC
    yr = {}
    for y in range(2028, 2033):
        sub = s[s.index.year == y]
        if len(sub):
            yr[str(y)] = round(float(sub.mean()), 4)
    rows.append(dict(name=name, ic=round(st["ic"],4), icir=round(st["icir"],3),
                     hit=round(st["hit"],3), n=st["n"], n_recent=len(recent),
                     ic_recent=round(r_ic,4) if np.isfinite(r_ic) else None,
                     icir_recent=round(r_icir,3) if np.isfinite(r_icir) else None,
                     gate=gate, yearly=yr))

print(f"{'factor':26s} {'IC10':>7s} {'ICIR':>6s} {'hit':>5s} {'n':>6s} {'IC_rcn':>7s} {'ICIR_rcn':>8s} {'gate':>5s}  yearly")
for r in rows:
    print(f"{r['name']:26s} {r['ic']:+.4f} {r['icir']:+.3f} {r['hit']:.2f} {r['n']:6d} "
          f"{r['ic_recent'] if r['ic_recent'] is not None else float('nan'):+.4f} "
          f"{r['icir_recent'] if r['icir_recent'] is not None else float('nan'):+.3f} "
          f"{'PASS' if r['gate'] else 'FAIL':>5s}  {r['yearly']}")
json.dump(dict(end=END, horizon=10, results=rows), open("scripts/miner_2_20320430_revalidation.json","w"), indent=1, default=str)
print("\nsaved scripts/miner_2_20320430_revalidation.json")