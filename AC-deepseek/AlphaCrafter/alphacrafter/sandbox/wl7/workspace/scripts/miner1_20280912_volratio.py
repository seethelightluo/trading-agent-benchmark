"""miner_1 2028-09-12 candidate: vol_ratio_5x60 (5d/60d realized vol ratio).

Hypothesis: assets whose short-term vol has recently expanded relative to their
medium-term vol are in stress (bear tape) and underperform / continue to be
pressured over 10d; alternatively vol-expansion marks capitulation bottoms.
Test signed IC across several (short,long) windows.
"""
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2028-09-11"
close = ms.load_close(END)
macro = ms.load_macro(END)
ret = close.pct_change()

for short, long in ((5, 60), (5, 40), (10, 60), (10, 40)):
    vs = ret.rolling(short).std()
    vl = ret.rolling(long).std()
    factor = vs / vl
    summ = ms.summarize(factor, close)
    h10 = summ[10]
    fwd = ms.forward_ret(close, 10)
    cov = ms.coverage_stats(factor, fwd)
    turn = ms.rank_turnover(factor, window=10)
    libs = ms.library_panel(close, macro)
    maxrho, pairs = ms.max_lib_corr(factor, libs)
    ic = ms.daily_ic(factor, fwd)
    yrs = {}
    for y, g in ic.groupby(ic.index.year):
        st = ms.ic_stats(g, 10)
        yrs[str(y)] = dict(ic=round(st["ic"], 4), icir=round(st["icir"], 4), n=st["n"])
    print(f"=== vol_ratio_{short}x{long} ===")
    print(f" h10 ic={h10['ic']:.4f} icir={h10['icir']:.4f} hit={h10['hit']:.3f} n={h10['n']}")
    print(f" decay:", {h: round(summ[h]['ic'], 4) for h in (1, 2, 3, 5, 10, 20)})
    print(f" coverage:", {k: round(v, 4) for k, v in cov.items()}, "turnover10:", round(turn, 3))
    print(f" max_lib_corr={maxrho:.3f} pairs={pairs}")
    print(f" per-year h10:", yrs)
    gate_ok = abs(h10["ic"]) >= ms.IC_GATE and abs(h10["icir"]) >= ms.ICIR_GATE
    print(f" GATE PASS: {gate_ok} (|ic|>={ms.IC_GATE}, |icir|>={ms.ICIR_GATE})")
