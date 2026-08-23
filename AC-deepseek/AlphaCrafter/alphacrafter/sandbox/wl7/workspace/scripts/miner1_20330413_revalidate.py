"""miner_1 revalidation of active library at visible_through 2033-04-08 (last completed trading day <= 2033-04-13)."""
import sys, json
sys.path.insert(0, "scripts")
from miner_shared import (
    master_calendar, load_close, load_macro, forward_ret, daily_ic,
    ic_stats, summarize, rank_turnover, coverage_stats, library_panel,
    max_lib_corr, IC_GATE, ICIR_GATE,
)

END = "2033-04-08"
close = load_close(END)
macro = load_macro(END)

lib = library_panel(close, macro)

print("=== Active library revalidation @ END=%s ===" % END)
print("dates:", close.shape[0], "assets:", close.shape[1])
print("cal_last:", master_calendar(END)[-1])

res = []
for name, panel in lib.items():
    ic = daily_ic(panel, forward_ret(close, 1))
    st = ic_stats(ic, 1)
    full = summarize(panel, close)
    turn = rank_turnover(panel)
    cov = coverage_stats(panel, forward_ret(close, 1))
    gate = (abs(st["ic"]) >= IC_GATE) and (abs(st["icir"]) >= ICIR_GATE)
    best, pairs = max_lib_corr(panel, {k: v for k, v in lib.items() if k != name})
    res.append(dict(
        name=name, ic=round(st["ic"],6), icir=round(st["icir"],6),
        hit=round(st["hit"],4), n=st["n"],
        ic_h10=round(full[10]["ic"],6), icir_h10=round(full[10]["icir"],6),
        turn=round(turn,4),
        covAD=round(cov["coverage_asset_days"],4), covD8=round(cov["coverage_dates_ge8"],4),
        maxrho=round(best,4), pairs=pairs, gate=bool(gate),
    ))
    print(f"{name:24s} IC(h1)={st['ic']:+.4f} ICIR(h1)={st['icir']:+.4f} hit={st['hit']:.3f} "
          f"n={st['n']} ic10={full[10]['ic']:+.4f} turn={turn:.2f} gate={gate} maxrho={best:.3f}")

json.dump(res, open("scripts/miner1_20330413_revalidation.json", "w"), indent=1)
print("saved: scripts/miner1_20330413_revalidation.json")