"""miner_2 periodic re-validation of the currently effective factor library.

Data visible through 2029-08-13 (previous completed trading day relative to
runtime current date 2029-08-14). Checks IC/ICIR stability vs admission gates
and flags drift/deprecation candidates.

Admission gates: abs(IC) >= 0.0070, abs(ICIR) >= 0.0840 (h=10).
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, daily_ic, ic_stats,
                          library_panel, ACTIVE_LIB)

END = "2029-08-13"
close = load_close(END)
macro = load_macro(END)
fwd10 = close.shift(-10) / close - 1.0

lib = library_panel(close, macro)

print(f"universe={close.shape[1]} assets, dates={len(close)} "
      f"({close.index[0].date()} -> {close.index[-1].date()})")
print(f"{'factor':28s} {'IC':>8s} {'ICIR':>7s} {'hit':>6s} {'n':>5s}  "
      f"{'2026IC':>8s} {'2027IC':>8s} {'2028IC':>8s} {'2029IC':>8s} {'2029ICIR':>8s}")

rows = []
for name, panel in lib.items():
    ic = daily_ic(panel, fwd10)
    st = ic_stats(ic, 10)
    ic26 = ic_stats(ic[ic.index.year == 2026], 10)
    ic27 = ic_stats(ic[ic.index.year == 2027], 10)
    ic28 = ic_stats(ic[ic.index.year == 2028], 10)
    ic29 = ic_stats(ic[ic.index.year == 2029], 10)
    flag = ""
    if abs(st["ic"]) < 0.0070 or abs(st["icir"]) < 0.0840:
        flag = " <-- BELOW GATE"
    if st["icir"] < 0:
        flag += " <-- NEG ICIR"
    print(f"{name:28s} {st['ic']:+8.4f} {st['icir']:+7.3f} {st['hit']:6.3f} "
          f"{st['n']:5d}  {ic26['ic']:+8.4f} {ic27['ic']:+8.4f} "
          f"{ic28['ic']:+8.4f} {ic29['ic']:+8.4f} {ic29['icir']:+8.3f}{flag}")
    rows.append(dict(factor=name, ic=st["ic"], icir=st["icir"], hit=st["hit"],
                     n=st["n"], ic26=ic26["ic"], ic27=ic27["ic"],
                     ic28=ic28["ic"], ic29=ic29["ic"], ic29_icir=ic29["icir"]))

print("\n=== recent 6-month rolling IC (h10) for drift monitor ===")
for name, panel in lib.items():
    ic = daily_ic(panel, fwd10)
    r = ic.rolling(126).mean().dropna()
    if len(r) == 0:
        continue
    print(f"{name:28s} last126d IC={r.iloc[-1]:+.4f}  min={r.min():+.4f} max={r.max():+.4f}")
