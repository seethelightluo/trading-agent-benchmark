"""miner_3 revalidation of active library factors, visible end 2032-02-26."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2032-02-26"

cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)

factors = ["rel_mom_20d_skip5", "beta_ew_60d", "corr_ew_60",
           "downside_vol_ratio_20", "kurt_20d_skip5", "max_ret_20d",
           "dxy_beta_cond_60x20", "eurusd_beta_cond_60x20"]

lib_panels = ms.library_panel(close, macro)
fwd = ms.forward_ret(close, 10)
ic_series = {}

print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")

def ic_stats_by_year(ic):
    s = ic.dropna()
    df = pd.DataFrame({"ic": s})
    df["year"] = s.index.year
    out = []
    for y, g in df.groupby("year"):
        m = g["ic"].mean(); sd = g["ic"].std(ddof=1)
        out.append((y, float(m), float(m / sd) if sd > 0 else np.nan, int(len(g))))
    return out

out = {"end": END, "horizon": 10, "results": {}}
print(f"{'factor':22s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s} {'maxrho':>7s}  GATE")
for name in factors:
    panel = lib_panels[name]
    ic = ms.daily_ic(panel, fwd)
    ic_series[name] = ic
    st = ms.ic_stats(ic, 10)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    mrho, pairs = ms.max_lib_corr(panel, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE) else "fail"
    print(f"{name:22s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    out["results"][name] = {
        "ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
        "coverage_dates_ge8": cov["coverage_dates_ge8"],
        "turnover": turn, "max_abs_library_correlation": round(mrho, 4),
        "corr_pairs": pairs,
        "per_year": [{"y": y, "ic": m, "icir": ir, "n": n} for y, m, ir, n in ic_stats_by_year(ic)],
    }

with open("scripts/miner3_20320227_revalidation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote scripts/miner3_20320227_revalidation.json")