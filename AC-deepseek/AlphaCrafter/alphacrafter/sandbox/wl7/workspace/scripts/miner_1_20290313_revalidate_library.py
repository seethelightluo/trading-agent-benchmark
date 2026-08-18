"""miner_1 periodic re-validation of the currently effective factor library.

Data visible through 2029-03-12 (previous completed trading day relative to
runtime current date 2029-03-13). Checks IC/ICIR stability, drift vs the
original warm-up admission metrics, and flags any factor whose IC has decayed
below threshold or ICIR turned negative (deprecation candidates).

Admission gates: abs(IC) >= 0.0070, abs(ICIR) >= 0.0840 (h=10).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, daily_ic, ic_stats,
                          library_panel, forward_ret)

END = "2029-03-12"
close = load_close(END)
macro = load_macro(END)
fwd10 = forward_ret(close, 10)

lib = library_panel(close, macro)

print(f"universe={close.shape[1]} assets, dates={len(close)} "
      f"({close.index[0].date()} -> {close.index[-1].date()})")
print(f"{'factor':28s} {'IC':>8s} {'ICIR':>7s} {'hit':>6s} {'n':>5s}  "
      f"{'2027IC':>8s} {'2028IC':>8s} {'2029IC':>8s} {'2029ICIR':>8s}")

rows = []
for name, panel in lib.items():
    ic = daily_ic(panel, fwd10)
    st = ic_stats(ic, 10)
    ic27 = ic_stats(ic[ic.index.year == 2027], 10)
    ic28 = ic_stats(ic[ic.index.year == 2028], 10)
    ic29 = ic_stats(ic[ic.index.year == 2029], 10)
    rows.append((name, st, ic27, ic28, ic29))
    flag = ""
    if abs(st["ic"]) < 0.0070 or abs(st["icir"]) < 0.0840:
        flag = " <-- BELOW GATE"
    if st["icir"] < 0:
        flag += " <-- NEG ICIR"
    print(f"{name:28s} {st['ic']:+8.4f} {st['icir']:+7.3f} {st['hit']:6.3f} "
          f"{st['n']:5d}  {ic27['ic']:+8.4f} {ic28['ic']:+8.4f} "
          f"{ic29['ic']:+8.4f} {ic29['icir']:+8.3f}{flag}")

print("\n=== recent 120d rolling IC (last 6 = 2028-10 .. 2029-03) ===")
for name, _, _, _, _ in rows:
    panel = lib[name]
    ic = daily_ic(panel, fwd10)
    roll = ic.rolling(120).mean().dropna()
    if len(roll) >= 6:
        print(f"{name:28s} " + " ".join(f"{v:+6.3f}" for v in roll.tail(6).values))

print("\n=== full-sample h10 pairwise corr vs library (provenance) ===")
for a, _, _, _, _ in rows:
    flat_a = lib[a].stack()
    best = (0.0, "")
    for b, _, _, _, _ in rows:
        if a == b:
            continue
        df = pd.concat([flat_a.rename("a"), lib[b].stack().rename("b")], axis=1).dropna()
        if len(df) < 30:
            continue
        r = df["a"].corr(df["b"])
        if abs(r) > abs(best[0]):
            best = (r, b)
    print(f"{a:28s} max|rho|={abs(best[0]):.3f} vs {best[1]}")
