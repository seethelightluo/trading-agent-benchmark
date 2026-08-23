"""miner_1 revalidation of active factor library at 2032-05-03 (visible through 2032-05-01).
Monitors only; does NOT touch live account, date.json, or account.json.
"""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, IC_GATE, ICIR_GATE,
                          max_lib_corr)

END = "2032-05-01"
close = load_close(END)
macro = load_macro(END)
lib_panels = library_panel(close, macro)
print(f"END={END}  n_dates={len(close)}  n_assets={close.shape[1]}")

names = list(lib_panels.keys())
rows = []
for name in names:
    f = lib_panels[name]
    fwd = forward_ret(close, 10)
    ic_s = daily_ic(f, fwd)
    st = ic_stats(ic_s, 10)
    cov = coverage_stats(f, fwd)
    turn = rank_turnover(f, 10)
    ic_ser = ic_s.dropna()
    recent = ic_ser[ic_ser.index >= (ic_ser.index.max() - np.timedelta64(365, 'D'))]
    r2 = ic_ser[ic_ser.index >= (ic_ser.index.max() - np.timedelta64(730, 'D'))]
    rows.append(dict(name=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                     ic_r=recent.mean() if len(recent) else np.nan,
                     icir_r=(recent.mean()/recent.std(ddof=1)) if len(recent) > 2 else np.nan,
                     n_r=len(recent),
                     ic_q=r2.mean() if len(r2) else np.nan,
                     icir_q=(r2.mean()/r2.std(ddof=1)) if len(r2) > 2 else np.nan,
                     n_q=len(r2),
                     covAD=cov["coverage_asset_days"], covD8=cov["coverage_dates_ge8"], turn=turn))

print("\n{:24s} {:>7s} {:>7s} {:>5s} {:>5s} | {:>7s} {:>7s} | {:>7s} {:>7s} | {:>6s} {:>5s} {:>6s}".format(
    "factor", "IC10", "ICIR10", "hit", "n", "IC_1y", "ICIR_1y", "IC_2y", "ICIR_2y", "covAD", "covD8", "turn"))
for r in rows:
    print("{:24s} {:+.4f} {:+.3f} {:5.2f} {:5d} | {:+.4f} {:+.3f} | {:+.4f} {:+.3f} | {:5.2f} {:5.2f} {:6.2f}".format(
        r["name"], r["ic"], r["icir"], r["hit"], r["n"], r["ic_r"], r["icir_r"], r["ic_q"], r["icir_q"],
        r["covAD"], r["covD8"], r["turn"]))

print("\nGATE (abs IC>=%.4f, abs ICIR>=%.4f, full-window h10):" % (IC_GATE, ICIR_GATE))
for r in rows:
    gate = abs(r["ic"]) >= IC_GATE and abs(r["icir"]) >= ICIR_GATE
    print("{:24s} IC={:+.4f} ICIR={:+.3f} -> {}".format(r["name"], r["ic"], r["icir"], "PASS" if gate else "FAIL"))

print("\nPer-year h10 IC:")
for name in names:
    f = lib_panels[name]
    fwd = forward_ret(close, 10)
    ic = daily_ic(f, fwd)
    out = []
    for yr in range(2030, 2033):
        sub = ic.loc[ic.index.year == yr]
        if len(sub) == 0:
            continue
        st = ic_stats(sub, 10)
        out.append("{:d}:{:+.3f}/{:+.2f}(n={:d})".format(yr, st["ic"], st["icir"], st["n"]))
    print("{:24s} {}".format(name, "  ".join(out)))

with open("scripts/miner1_20320503_revalidation.json", "w") as fo:
    json.dump(dict(end=END, horizon=10, results={
        r["name"]: {k: r[k] for k in ("ic", "icir", "hit", "n", "ic_r", "icir_r", "n_r", "ic_q", "icir_q", "n_q", "covAD", "covD8", "turn")}
        for r in rows}), fo, indent=1, default=str)
print("\nsaved scripts/miner1_20320503_revalidation.json")