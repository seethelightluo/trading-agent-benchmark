"""miner_2 revalidation of active factor library up through visible_through 2033-03-01."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np
from miner_shared import (
    load_close, load_macro, forward_ret, daily_ic, ic_stats, summarize,
    rank_turnover, coverage_stats, library_panel, max_lib_corr,
    IC_GATE, ICIR_GATE, ACTIVE_LIB,
)

END = "2033-03-01"
H = 10
close = load_close(END)
macro = load_macro(END)
ret = close.pct_change()
fwd = forward_ret(close, H)
lib = library_panel(close, macro)

print(f"=== Active library revalidation END={END} horizon={H} ===")
print("dates:", close.shape[0], "assets:", close.shape[1])

results = {}
for name, panel in lib.items():
    ic = daily_ic(panel, fwd)
    st = ic_stats(ic, H)
    full = summarize(panel, close)
    turn = rank_turnover(panel)
    cov = coverage_stats(panel, fwd)
    gate = (abs(st["ic"]) >= IC_GATE) and (abs(st["icir"]) >= ICIR_GATE)
    best, pairs = max_lib_corr(panel, {k: v for k, v in lib.items() if k != name})
    s = ic.dropna()
    def recent(dt):
        sub = s[s.index >= dt]
        if len(sub)==0: return (np.nan, np.nan, 0)
        m=sub.mean(); sd=sub.std(ddof=1)
        return (float(m), float(m/sd) if sd>0 else np.nan, float((sub>0).mean()))
    r6m=recent("2032-09-01"); r1y=recent("2032-03-01"); r2y=recent("2031-03-01")
    py = {}
    for y, g in s.groupby(s.index.year):
        m=g.mean(); sd=g.std(ddof=1)
        py[y]=round(float(m),4)
    results[name]=dict(ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                       gate=bool(gate), turn=turn, covAD=cov["coverage_asset_days"],
                       covD8=cov["coverage_dates_ge8"], maxrho=best,
                       r6m=r6m, r1y=r1y, r2y=r2y, per_year=py)
    print(f"\n{name}: IC {st['ic']:+.4f} ICIR {st['icir']:+.3f} hit {st['hit']:.3f} n {st['n']} gate={gate}")
    print(f"  covAD {cov['coverage_asset_days']:.3f} dates_ge8 {cov['coverage_dates_ge8']:.3f} turn {turn:.2f} maxrho {best:.3f}")
    print(f"  r6m IC {r6m[0]:+.4f} | r1y {r1y[0]:+.4f} | r2y {r2y[0]:+.4f}")
    print("  per_year:", py)

json.dump(dict(end=END, horizon=H, results=results),
          open("scripts/miner2_20330217_lib_revalidate.json","w"), indent=1, default=str)
print("\nsaved scripts/miner2_20330217_lib_revalidate.json")