"""miner_2 revalidation of active factor library through visible 2033-03-29."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
import miner_shared as M

END = "2033-03-29"
H = 10
close = M.load_close(END)
macro = M.load_macro(END)
ret = close.pct_change()
fwd = M.forward_ret(close, H)
lib = M.library_panel(close, macro)

print(f"=== Active library revalidation END={END} horizon={H} ===")
print("dates:", close.shape[0], "assets:", close.shape[1])

results = {}
for name, panel in lib.items():
    ic = M.daily_ic(panel, fwd)
    st = M.ic_stats(ic, H)
    full = M.summarize(panel, close)
    turn = M.rank_turnover(panel)
    cov = M.coverage_stats(panel, fwd)
    gate = (abs(st["ic"]) >= M.IC_GATE) and (abs(st["icir"]) >= M.ICIR_GATE)
    best, pairs = M.max_lib_corr(panel, {k: v for k, v in lib.items() if k != name})
    s = ic.dropna()
    def recent(dt):
        sub = s[s.index >= dt]
        if len(sub) == 0:
            return (np.nan, np.nan, 0)
        m = sub.mean(); sd = sub.std(ddof=1)
        return (float(m), float(m/sd) if sd > 0 else np.nan, float((sub > 0).mean()))
    r6m = recent("2032-09-29"); r1y = recent("2032-03-29")
    py = {}
    for y, g in s.groupby(s.index.year):
        m = g.mean(); sd = g.std(ddof=1)
        py[int(y)] = round(float(m), 4)
    results[name] = dict(ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                         gate=bool(gate), turn=turn, covAD=cov["coverage_asset_days"],
                         covD8=cov["coverage_dates_ge8"], maxrho=best,
                         r6m=r6m, r1y=r1y, per_year=py)
    print(f"\n{name}: IC {st['ic']:+.4f} ICIR {st['icir']:+.3f} hit {st['hit']:.3f} n {st['n']} gate={gate}")
    print(f"  covAD {cov['coverage_asset_days']:.3f} dates_ge8 {cov['coverage_dates_ge8']:.3f} turn {turn:.2f} maxrho {best:.3f}")
    print(f"  r6m IC {r6m[0]:+.4f} | r1y {r1y[0]:+.4f}")
    print("  per_year:", py)

json.dump(dict(end=END, horizon=H, results=results),
          open("scripts/miner2_20330330_lib_revalidate.json", "w"), indent=1, default=str)
print("\nsaved scripts/miner2_20330330_lib_revalidate.json")