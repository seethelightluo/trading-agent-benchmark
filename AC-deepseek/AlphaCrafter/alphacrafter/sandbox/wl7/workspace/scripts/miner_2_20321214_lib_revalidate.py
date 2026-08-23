"""miner_2 revalidation of active 8-factor library + candidate screening, visible end 2032-12-13."""
import sys, json
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from miner_shared import (ASSETS, load_close, load_macro, forward_ret, daily_ic, ic_stats,
                          coverage_stats, library_panel, max_lib_corr, rank_turnover, summarize,
                          ACTIVE_LIB)

END = "2032-12-13"
close = load_close(END); macro = load_macro(END)
ret = close.pct_change()
fwd10 = forward_ret(close, 10)
lib = library_panel(close, macro)

print("="*70)
print("ACTIVE LIBRARY REVALIDATION  end", END, " horizon 10")
print("="*70)
results = {}
for name, (fn, kw) in ACTIVE_LIB.items():
    if name == "corr_ew_60":
        f = fn(ret, **kw)
    elif name == "dxy_beta_cond_60x20":
        f = fn(close, macro["DXY"], **kw)
    elif name == "eurusd_beta_cond_60x20":
        f = fn(close, macro["EURUSD"], **kw)
    else:
        f = fn(close, **kw)
    ic = daily_ic(f, fwd10)
    st = ic_stats(ic, 10)
    cov = coverage_stats(f, fwd10)
    turn = rank_turnover(f, 10)
    mrho, pairs = max_lib_corr(f, lib)
    s = ic.dropna()
    def recent(dt):
        sub = s[s.index >= dt]
        if len(sub) == 0: return (np.nan, np.nan, 0)
        m = sub.mean(); sd = sub.std(ddof=1)
        return (float(m), float(m/sd) if sd>0 else np.nan, float((sub>0).mean()))
    r2y = recent("2030-12-01"); r1y = recent("2031-12-01"); r6m = recent("2032-06-01")
    per_year = []
    for y, g in s.groupby(s.index.year):
        m = g.mean(); sd = g.std(ddof=1)
        per_year.append((y, float(m), float(m/sd) if sd>0 else np.nan, int(len(g))))
    results[name] = dict(ic=st["ic"], icir=st["icir"], hit=st["hit"], n=st["n"],
                         cov_asset_days=cov["coverage_asset_days"], cov_dates_ge8=cov["coverage_dates_ge8"],
                         turnover=turn, max_abs_lib_correlation=mrho,
                         r2y=r2y, r1y=r1y, r6m=r6m, per_year=per_year)
    okF = abs(st["ic"]) >= 0.0070 and abs(st["icir"]) >= 0.0840
    print(f"\n{name}: IC {st['ic']:+.4f} ICIR {st['icir']:+.3f} hit {st['hit']:.3f} n {st['n']} gate_full={okF}")
    print(f"  cov_asset_days {cov['coverage_asset_days']:.3f} dates_ge8 {cov['coverage_dates_ge8']:.3f} turnover {turn:.2f} maxlib {mrho:.3f}")
    print(f"  recent  6m IC {r6m[0]:+.4f}({r6m[1]:+.2f})hit{r6m[2]:.2f} | 1y {r1y[0]:+.4f}({r1y[1]:+.2f}) | 2y {r2y[0]:+.4f}({r2y[1]:+.2f})")
    py = {y: round(m,3) for y,m,ir,n in per_year}
    print("  per_year:", py)

json.dump(dict(end=END, horizon=10, results=results), open("scripts/miner_2_20321214_lib_revalidate.json","w"), indent=1, default=str)
print("\nsaved scripts/miner_2_20321214_lib_revalidate.json")