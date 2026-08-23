"""miner_3 revalidation of active library factors + new candidate, visible end 2032-07-23."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2032-07-23"

cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)

ret = close.pct_change()
fwd = ms.forward_ret(close, 10)

lib_panels = ms.library_panel(close, macro)
factors = list(lib_panels.keys())

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
        "turnover": turn, "max_abs_library_correlation": round(mrho, 4),
        "corr_pairs": pairs,
        "per_year": [{"y": y, "ic": m, "icir": ir, "n": n} for y, m, ir, n in ic_stats_by_year(ic)],
    }


# ---- New candidate exploration: short-horizon signal-to-noise momentum ----
def cand_tstat_mom(close, win, skip):
    """t-stat of trailing mean daily return: mean(win)/std(win), skip recent news."""
    r = close.pct_change()
    r = r.shift(skip)
    mu = r.rolling(win, min_periods=win // 2).mean()
    sd = r.rolling(win, min_periods=win // 2).std()
    return mu.divide(sd.add(1e-9), axis=0)


def cand_vol_regime(close, fast=20, slow=60):
    """Volatility regime change: 20d/60d realized vol ratio shift over 20d."""
    r = close.pct_change()
    vol = r.rolling(fast).std().divide(r.rolling(slow).std().add(1e-9), axis=0)
    return vol - vol.shift(fast)


print("\n--- Candidate exploration (horizon 10) ---")
cands = {}
for win, skip in [(10, 3), (20, 5)]:
    cands[f"tstat_mom_{win}x{skip}"] = cand_tstat_mom(close, win, skip)
cands["vol_regime_shift_20x60"] = cand_vol_regime(close)

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
        "turnover": turn, "max_abs_library_correlation": round(mrho, 4),
        "corr_pairs": pairs,
        "per_year": [{"y": y, "ic": m, "icir": ir, "n": n} for y, m, ir, n in ic_stats_by_year(ic)],
    }

with open("scripts/miner3_20320726_revalidation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote scripts/miner3_20320726_revalidation.json")
