"""miner_2 periodic re-validation of the currently effective factor library.

Data visible through 2031-02-24 (previous completed trading day relative to
runtime current date 2031-02-25). Checks IC/ICIR stability vs admission gates
and flags drift/deprecation candidates.

Admission gates: abs(IC) >= 0.0070, abs(ICIR) >= 0.0840 (h=10, 15-asset universe).
Includes recent 500d/250d windows for drift assessment + per-year IC.
"""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          library_panel, ACTIVE_LIB)

END = "2031-02-24"
close = load_close(END)
macro = load_macro(END)
fwd10 = forward_ret(close, 10)

lib = library_panel(close, macro)

print(f"universe={close.shape[1]} assets, dates={len(close)} "
      f"({close.index[0].date()} -> {close.index[-1].date()})")

names = list(ACTIVE_LIB.keys())
rows = []
print(f"\n{'factor':26s} {'IC10':>8s} {'ICIR10':>7s} {'hit':>6s} {'n':>5s} | "
      f"{'IC_r500':>8s} {'ICIR_r500':>9s} | {'IC_q250':>7s} {'ICIR_q250':>8s} | {'turn':>6s}")
for name in names:
    f = lib[name]
    ic = daily_ic(f, fwd10)
    st = ic_stats(ic, 10)
    f_r = f.tail(500)
    ic_r = daily_ic(f_r, forward_ret(close, 10).reindex(f_r.index))
    st_r = ic_stats(ic_r, 10)
    f_q = f.tail(250)
    ic_q = daily_ic(f_q, forward_ret(close, 10).reindex(f_q.index))
    st_q = ic_stats(ic_q, 10)
    rows.append(dict(factor=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                     ic_r=st_r["ic"], icir_r=st_r["icir"], n_r=st_r["n"],
                     ic_q=st_q["ic"], icir_q=st_q["icir"], n_q=st_q["n"]))
    print(f"{name:26s} {st['ic']:+8.4f} {st['icir']:+7.3f} {st['hit']:6.3f} {st['n']:5d} | "
          f"{st_r['ic']:+8.4f} {st_r['icir']:+7.3f} | {st_q['ic']:+7.4f} {st_q['icir']:+7.3f} | "
          f"{ic_r.iloc[-1] if False else '':6s}")

print("\nGATE check (abs IC>=0.0070, abs ICIR>=0.0840, full-window h10):")
for r in rows:
    gate_pass = abs(r["ic"]) >= 0.0070 and abs(r["icir"]) >= 0.0840
    print(f"{r['name']:26s} IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} -> {'PASS' if gate_pass else 'FAIL'}")

print("\nPer-year h10 IC (sign as stored):")
for name in names:
    f = lib[name]
    ic = daily_ic(f, fwd10)
    out = []
    for yr in range(2026, 2032):
        sub = ic.loc[ic.index.year == yr]
        st = ic_stats(sub, 10)
        out.append(f"{yr}:{st['ic']:+.3f}/{st['icir']:+.2f}(n={st['n']})")
    print(f"{name:26s} " + "  ".join(out))

print("\n=== most recent 126d rolling IC (h10) drift monitor ===")
for name in names:
    ic = daily_ic(lib[name], fwd10)
    r = ic.rolling(126).mean().dropna()
    if len(r) == 0:
        continue
    print(f"{name:26s} last126d IC={r.iloc[-1]:+.4f}  min={r.min():+.4f} max={r.max():+.4f}")

print("\n=== most recent 63d (quarter) IC ===")
for name in names:
    ic = daily_ic(lib[name], fwd10)
    r = ic.rolling(63).mean().dropna()
    if len(r) == 0:
        continue
    print(f"{name:26s} last63d IC={r.iloc[-1]:+.4f}  min={r.min():+.4f} max={r.max():+.4f}")