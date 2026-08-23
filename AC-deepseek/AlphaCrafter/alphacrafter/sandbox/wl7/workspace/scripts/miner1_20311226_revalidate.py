"""miner_1 revalidation of active factor library at 2031-12-26 (visible through 2031-12-11).
Only runs validation/monitoring; does NOT touch live account, date.json, or account.json."""
import sys, json
sys.path.insert(0, "scripts")
from miner_shared import (load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          rank_turnover, coverage_stats, library_panel, IC_GATE, ICIR_GATE)

END = "2031-12-11"
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
    f_r = f.tail(500)
    ic_r = daily_ic(f_r, forward_ret(close, 10).reindex(f_r.index))
    st_r = ic_stats(ic_r, 10)
    f_q = f.tail(250)
    ic_q = daily_ic(f_q, forward_ret(close, 10).reindex(f_q.index))
    st_q = ic_stats(ic_q, 10)
    rows.append(dict(name=name, ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                     ic_r=st_r["ic"], icir_r=st_r["icir"], n_r=st_r["n"],
                     ic_q=st_q["ic"], icir_q=st_q["icir"], n_q=st_q["n"],
                     covAD=cov["coverage_asset_days"], covD8=cov["coverage_dates_ge8"], turn=turn))

print("\n{:26s} {:>7s} {:>7s} {:>5s} {:>5s} | {:>7s} {:>7s} | {:>7s} {:>7s} | {:>6s} {:>5s} {:>6s}".format(
    "factor", "IC10", "ICIR10", "hit", "n", "IC_r", "ICIR_r", "IC_q", "ICIR_q", "covAD", "covD8", "turn"))
for r in rows:
    print("{:26s} {:+.4f} {:+.3f} {:5.2f} {:5d} | {:+.4f} {:+.3f} | {:+.4f} {:+.3f} | {:5.2f} {:5.2f} {:6.2f}".format(
        r["name"], r["ic"], r["icir"], r["hit"], r["n"], r["ic_r"], r["icir_r"],
        r["ic_q"], r["icir_q"], r["covAD"], r["covD8"], r["turn"]))

print("\nGATE (abs IC>=%.4f, abs ICIR>=%.4f, full-window h10):" % (IC_GATE, ICIR_GATE))
for r in rows:
    gate = abs(r["ic"]) >= IC_GATE and abs(r["icir"]) >= ICIR_GATE
    print("{:26s} IC={:+.4f} ICIR={:+.3f} -> {}".format(r["name"], r["ic"], r["icir"], "PASS" if gate else "FAIL"))

print("\nPer-year h10 IC:")
for name in names:
    f = lib_panels[name]
    fwd = forward_ret(close, 10)
    ic = daily_ic(f, fwd)
    out = []
    for yr in range(2029, 2033):
        sub = ic.loc[ic.index.year == yr]
        if len(sub) == 0:
            continue
        st = ic_stats(sub, 10)
        out.append("{:d}:{:+.3f}/{:+.2f}(n={:d})".format(yr, st["ic"], st["icir"], st["n"]))
    print("{:26s} {}".format(name, "  ".join(out)))

# recent drift: fwd-h10 on last ~1 year (approx 250 trading days)
print("\nRecent 1y & 2y IC markers (raw IC only):")
for r in rows:
    print("{:26s} ic_1y={:+.4f}(n={:d})  ic_2y={:+.4f}(n={:d})".format(
        r["name"], r["ic_r"], r["n_r"], r["ic_q"], r["n_q"]))

with open("scripts/miner1_20311226_revalidation.json", "w") as fo:
    json.dump(dict(end=END, horizon=10, results={
        r["name"]: {k: r[k] for k in ("ic","icir","hit","n","ic_r","icir_r","n_r","ic_q","icir_q","n_q","covAD","covD8","turn")}
        for r in rows}), fo, indent=1, default=str)
print("\nsaved scripts/miner1_20311226_revalidation.json")