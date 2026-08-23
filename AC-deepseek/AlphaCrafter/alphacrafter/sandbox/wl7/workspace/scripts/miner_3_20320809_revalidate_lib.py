"""miner_3 revalidation of active library factors, visible end 2032-08-09.

Recomputes ensemble + eurusd factors on master 15-asset universe, per-date
cross-sectional Spearman IC vs h=10 forward returns, IC/ICIR/hit, per-year
breakdown, coverage, turnover, decay by horizon, and max abs library correlation
excluding self. No lookahead. Gates: |IC|>=0.0070, |ICIR|>=0.0840.
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2032-08-09"

cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)

factors = ["rel_mom_20d_skip5", "beta_ew_60d", "corr_ew_60",
           "downside_vol_ratio_20", "kurt_20d_skip5", "max_ret_20d",
           "dxy_beta_cond_60x20", "eurusd_beta_cond_60x20"]

lib_panels = ms.library_panel(close, macro)
fwd = ms.forward_ret(close, 10)
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


def max_lib_corr_excl_self(candidate_name, lib_panels):
    cand = lib_panels[candidate_name]
    flat = cand.stack()
    best = 0.0; pairs = {}
    for name, p in lib_panels.items():
        if name == candidate_name:
            continue
        pflat = p.reindex(cand.index).stack()
        df = pd.concat([flat.rename("f"), pflat.rename("p")], axis=1).dropna()
        if len(df) < 30:
            continue
        rho = float(df["f"].corr(df["p"]))
        pairs[name] = round(rho, 4)
        if abs(rho) > best:
            best = abs(rho)
    return best, pairs


def decay_table(panel):
    out = {}
    for h in (1, 2, 3, 5, 10, 20):
        fw = ms.forward_ret(close, h)
        ic = ms.daily_ic(panel, fw)
        st = ms.ic_stats(ic, h)
        out[str(h)] = round(st["ic"], 4)
    return out


out = {"end": END, "horizon": 10, "results": {}}
print(f"{'factor':22s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s} {'maxrho':>7s}  GATE")
for name in factors:
    panel = lib_panels[name]
    ic = ms.daily_ic(panel, fwd)
    st = ms.ic_stats(ic, 10)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    mrho, pairs = max_lib_corr_excl_self(name, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE) else "fail"
    print(f"{name:22s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    out["results"][name] = {
        "ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
        "coverage_asset_days": cov["coverage_asset_days"],
        "coverage_dates_ge8": cov["coverage_dates_ge8"],
        "turnover_10d": turn, "max_abs_library_correlation": round(mrho, 4),
        "corr_pairs_excl_self": pairs,
        "decay_ic_by_horizon": decay_table(panel),
        "per_year": [{"y": y, "ic": m, "icir": ir, "n": n} for y, m, ir, n in ic_stats_by_year(ic)],
    }

print("\n--- recent 1y / 2y ---")
for name in factors:
    panel = lib_panels[name]
    ic = ms.daily_ic(panel, fwd).dropna()
    for lab, nd in (("1y", 365), ("2y", 730)):
        s = ic[ic.index >= ic.index.max() - np.timedelta64(nd, "D")]
        if len(s) == 0:
            continue
        m = s.mean(); sd = s.std(ddof=1); ir = m / sd if sd > 0 else np.nan
        print(f"{name:22s} {lab}: IC {m:+.4f} ICIR {ir:+.3f} hit {(s>0).mean():.2f} n {len(s)}")
        out["results"][name][f"recent_{lab}"] = {"ic": float(m), "icir": float(ir),
                                                 "hit": float((s > 0).mean()), "n": int(len(s))}

with open("scripts/miner3_20320809_revalidation.json", "w") as h:
    json.dump(out, h, indent=2)
print("\nWrote scripts/miner3_20320809_revalidation.json")