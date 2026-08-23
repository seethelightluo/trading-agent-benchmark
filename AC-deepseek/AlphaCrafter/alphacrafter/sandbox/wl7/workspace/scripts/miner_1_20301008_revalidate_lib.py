"""miner_1 revalidation of active factor library at 2030-09-10 (visible through 2030-09-09)."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          library_panel, coverage_stats, rank_turnover)

END = "2030-10-07"
close = load_close(END)
macro = load_macro(END)
print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}")

names = ["rel_mom_20d_skip5", "beta_ew_60d", "corr_ew_60", "downside_vol_ratio_20",
         "dxy_beta_cond_60x20", "eurusd_beta_cond_60x20", "kurt_20d_skip5", "max_ret_20d"]
lib_panels = library_panel(close, macro)
fwd = forward_ret(close, 10)

rows = []
for name in names:
    f = lib_panels[name]
    st = daily_ic(f, fwd).pipe(ic_stats, 10)
    ic_r = daily_ic(f.iloc[-500:], forward_ret(close.iloc[-500:], 10)).pipe(ic_stats, 10)
    ic_q = daily_ic(f.iloc[-250:], forward_ret(close.iloc[-250:], 10)).pipe(ic_stats, 10)
    ic_m = daily_ic(f.iloc[-90:], forward_ret(close.iloc[-90:], 10)).pipe(ic_stats, 10)
    cov = coverage_stats(f, fwd)
    turn = rank_turnover(f, 10)
    rows.append(dict(name=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                     ic_r=ic_r["ic"], icir_r=ic_r["icir"], n_r=ic_r["n"],
                     ic_q=ic_q["ic"], icir_q=ic_q["icir"], n_q=ic_q["n"],
                     ic_m=ic_m["ic"], icir_m=ic_m["icir"], n_m=ic_m["n"],
                     covAD=cov["coverage_asset_days"], covD8=cov["coverage_dates_ge8"], turn=turn))

print(f"\n{'factor':26s} {'IC10':>7s} {'ICIR10':>7s} {'hit':>5s} {'n':>5s} | {'IC_r':>7s} {'ICIR_r':>7s} {'n_r':>4s} | {'IC_q':>7s} {'ICIR_q':>7s} {'n_q':>4s} | {'IC_m':>7s} {'ICIR_m':>7s} {'n_m':>4s} | {'covAD':>6s} {'covD8':>5s} {'turn':>6s}")
for r in rows:
    print(f"{r['name']:26s} {r['ic']:7.4f} {r['icir']:7.3f} {r['hit']:5.2f} {r['n']:5d} | "
          f"{r['ic_r']:7.4f} {r['icir_r']:7.3f} {r['n_r']:4d} | "
          f"{r['ic_q']:7.4f} {r['icir_q']:7.3f} {r['n_q']:4d} | "
          f"{r['ic_m']:7.4f} {r['icir_m']:7.3f} {r['n_m']:4d} | "
          f"{r['covAD']:6.2f} {r['covD8']:5.2f} {r['turn']:6.2f}")

print("\nGATE check (abs IC>=0.0070, abs ICIR>=0.0840, full-window h10):")
for r in rows:
    gate_pass = abs(r["ic"]) >= 0.0070 and abs(r["icir"]) >= 0.0840
    print(f"{r['name']:26s} IC={r['ic']:+.4f} ICIR={r['icir']:+.3f} -> {'PASS' if gate_pass else 'FAIL'}")

print("\nPer-year h10 IC (sign as stored):")
for name in names:
    f = lib_panels[name]
    ic = daily_ic(f, fwd)
    out = []
    for yr in range(2026, 2031):
        sub = ic.loc[ic.index.year == yr]
        st = ic_stats(sub, 10)
        out.append(f"{yr}:{st['ic']:+.3f}/{st['icir']:+.2f}(n={st['n']})")
    print(f"{name:26s} " + "  ".join(out))

print("\nMacro/regime snapshot (last 21 rows of visible window):")
m = macro.tail(21)
for c in macro.columns:
    v = m[c]
    chg = (v.iloc[-1] / v.iloc[0] - 1) * 100
    print(f"{c:8s} last={v.iloc[-1]:.3f} 21d_chg={chg:+.2f}%")

print("\nLast-21d asset returns (visible window):")
cl = close.tail(21)
for a in close.columns:
    chg = (cl[a].iloc[-1] / cl[a].iloc[0] - 1) * 100
    print(f"{a:10s} {chg:+.2f}%  last_close={cl[a].iloc[-1]:.2f}")
