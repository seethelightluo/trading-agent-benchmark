"""miner_1 periodic re-validation of the currently effective factor library.

Data visible through 2028-12-04 (previous completed trading day relative to
runtime current date 2028-12-05). Checks IC/ICIR stability, drift vs the
original warm-up admission metrics, and flags any factor whose IC has decayed
below threshold or ICIR turned negative (deprecation candidates).

Admission gates: abs(IC) >= 0.0070, abs(ICIR) >= 0.0840 (h=10).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, daily_ic, ic_stats,
                          library_panel)

END = "2028-12-04"
close = load_close(END)
macro = load_macro(END)
fwd10 = close.shift(-10) / close - 1.0

lib = library_panel(close, macro)

print(f"universe={close.shape[1]} assets, dates={len(close)} "
      f"({close.index[0].date()} -> {close.index[-1].date()})")
print(f"{'factor':28s} {'IC':>8s} {'ICIR':>7s} {'hit':>6s} {'n':>5s}  "
      f"{'2027IC':>8s} {'2028IC':>8s} {'2028ICIR':>7s}")

rows = []
for name, panel in lib.items():
    ic = daily_ic(panel, fwd10)
    st = ic_stats(ic, 10)
    # drift windows
    ic27 = ic_stats(ic[ic.index.year == 2027], 10)
    ic28 = ic_stats(ic[ic.index.year == 2028], 10)
    rows.append((name, st, ic27, ic28))
    flag = ""
    if abs(st["ic"]) < 0.0070 or abs(st["icir"]) < 0.0840:
        flag = " <-- BELOW GATE"
    if st["icir"] < 0:
        flag += " <-- NEG ICIR"
    print(f"{name:28s} {st['ic']:+8.4f} {st['icir']:+7.3f} {st['hit']:6.3f} "
          f"{st['n']:5d}  {ic27['ic']:+8.4f} {ic28['ic']:+8.4f} "
          f"{ic28['icir']:+7.3f}{flag}")

print("\n=== full-sample h10 pairwise corr vs library (provenance) ===")
for a, (sa, _, _) in rows:
    flat_a = lib[a].stack()
    best = (0.0, "")
    for b, (sb, _, _) in rows:
        if a == b:
            continue
        df = pd.concat([flat_a.rename("a"), lib[b].stack().rename("b")], axis=1).dropna()
        if len(df) < 30:
            continue
        r = df["a"].corr(df["b"])
        if abs(r) > abs(best[0]):
            best = (r, b)
    print(f"{a:28s} max|rho|={abs(best[0]):.3f} vs {best[1]}")
