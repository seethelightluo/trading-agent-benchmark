"""miner_3 periodic revalidation of active library factors, visible end 2032-12-27."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2032-12-27"
cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)
fwd = ms.forward_ret(close, 10)
lib_panels = ms.library_panel(close, macro)
print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")


def max_lib_corr_excl_self(candidate_name, lib_panels2):
    cand = lib_panels2[candidate_name]
    flat = cand.stack(); best = 0.0; pairs = {}
    for name, p in lib_panels2.items():
        if name == candidate_name:
            continue
        pflat = p.reindex(cand.index).stack()
        df = pd.concat([flat.rename("f"), pflat.rename("p")], axis=1).dropna()
        if len(df) < 30:
            continue
        rho = float(df["f"].corr(df["p"])); pairs[name] = round(rho, 4)
        if abs(rho) > best:
            best = abs(rho)
    return best, pairs


def decay_table(panel):
    out = {}
    for h in (1, 2, 3, 5, 10, 20):
        fw = ms.forward_ret(close, h)
        out[str(h)] = round(ms.ic_stats(ms.daily_ic(panel, fw), h)["ic"], 4)
    return out


factors = ["rel_mom_20d_skip5", "beta_ew_60d", "corr_ew_60",
           "downside_vol_ratio_20", "kurt_20d_skip5", "max_ret_20d",
           "dxy_beta_cond_60x20", "eurusd_beta_cond_60x20"]

out = {"end": END, "horizon": 10, "results": {}}
print(f"{'factor':22s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s} {'maxrho':>7s}  GATE")
for name in factors:
    panel = lib_panels[name]
    st = ms.ic_stats(ms.daily_ic(panel, fwd), 10)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    mrho, pairs = max_lib_corr_excl_self(name, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE) else "fail"
    print(f"{name:22s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    ic = ms.daily_ic(panel, fwd).dropna()
    out["results"][name] = {"ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
                            "coverage_dates_ge8": cov["coverage_dates_ge8"],
                            "turnover_10d": turn, "max_abs_library_correlation": round(mrho, 4),
                            "decay_ic_by_horizon": decay_table(panel), "library_pairs": pairs}
    for lab, nd in (("1y", 365), ("6m", 183), ("3m", 91)):
        s = ic[ic.index >= ic.index.max() - np.timedelta64(nd, "D")]
        if len(s) == 0:
            continue
        m = s.mean(); sd = s.std(ddof=1); ir = m / sd if sd > 0 else np.nan
        print(f"    {lab}: IC {m:+.4f} ICIR {ir:+.3f} hit {(s>0).mean():.2f} n {len(s)}")
        out["results"][name][f"recent_{lab}"] = {"ic": float(m), "icir": float(ir),
                                                 "hit": float((s > 0).mean()), "n": int(len(s))}

with open("scripts/miner3_20321228_revalidation.json", "w") as h:
    json.dump(out, h, indent=2)
print("\nWrote scripts/miner3_20321228_revalidation.json")