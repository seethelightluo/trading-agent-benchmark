"""miner_3 revalidation of active library factors + new candidate, visible end 2032-12-27."""
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

ret = close.pct_change()
fwd = ms.forward_ret(close, 10)

lib_panels = ms.library_panel(close, macro)
factors = list(lib_panels.keys())

print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")


def ic_series_names(ic):
    s = ic.dropna()
    df = pd.DataFrame({"ic": s})
    df["year"] = s.index.year
    out = []
    for y, g in df.groupby("year"):
        m = g["ic"].mean(); sd = g["ic"].std(ddof=1)
        out.append((y, float(m), float(m / sd) if sd > 0 else np.nan, int(len(g))))
    return out


out = {"end": END, "horizon": 10, "results": {}}
print(f"{'factor':24s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s} {'maxrho':>7s}  GATE")
for name in factors:
    panel = lib_panels[name]
    ic = ms.daily_ic(panel, fwd)
    st = ms.ic_stats(ic, 10)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    mrho, pairs = ms.max_lib_corr(panel, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE) else "fail"
    print(f"{name:24s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    out["results"][name] = {
        "ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
        "coverage_dates_ge8": cov["coverage_dates_ge8"],
        "turnover_10d": turn, "max_abs_library_correlation": round(mrho, 4),
        "corr_pairs": {k: round(v, 4) for k, v in pairs.items()},
        "decay_ic_by_horizon": {str(h): round(ms.ic_stats(ms.daily_ic(panel, ms.forward_ret(close, h)), h)["ic"], 4) for h in (1, 2, 3, 5, 10, 20)},
        "per_year": [{"y": y, "ic": m, "icir": ir, "n": n} for y, m, ir, n in ic_series_names(ic)],
    }

# ---- New candidate: macro beta conditioning on USDCNY momentum ----
# Analogous to dxy_beta_cond / eurusd_beta_cond but using USDCNY.
def cand_cny_beta_cond(close, cny, beta_win=60, cond_win=20, min_periods=30):
    r = close.pct_change()
    fx_r = cny.pct_change()
    cov = r.rolling(beta_win, min_periods=min_periods).cov(fx_r)
    var = fx_r.rolling(beta_win, min_periods=min_periods).var()
    beta = cov.divide(var, axis=0)
    fx_mom = cny / cny.shift(cond_win) - 1.0
    return beta.multiply(fx_mom, axis=0)

print("\n--- Candidate exploration (horizon 10) ---")
cands = {"cny_beta_cond_60x20": cand_cny_beta_cond(close, macro["USDCNY"])}

for name, c in cands.items():
    ic = ms.daily_ic(c, fwd)
    st = ms.ic_stats(ic, 10)
    cov = ms.coverage_stats(c, fwd)
    turn = ms.rank_turnover(c, window=10)
    mrho, pairs = ms.max_lib_corr(c, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE) else "fail"
    print(f"{name:24s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    out["results"][name] = {
        "ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
        "coverage_dates_ge8": cov["coverage_dates_ge8"],
        "turnover_10d": turn, "max_abs_library_correlation": round(mrho, 4),
        "corr_pairs": {k: round(v, 4) for k, v in pairs.items()},
        "decay_ic_by_horizon": {str(h): round(ms.ic_stats(ms.daily_ic(c, ms.forward_ret(close, h)), h)["ic"], 4) for h in (1, 2, 3, 5, 10, 20)},
        "per_year": [{"y": y, "ic": m, "icir": ir, "n": n} for y, m, ir, n in ic_series_names(ic)],
    }

with open("scripts/miner3_20321228_revalidation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote scripts/miner3_20321228_revalidation.json")