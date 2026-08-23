"""miner_3 fresh revalidation (through 2033-02-16) + new candidate exploration."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2033-02-16"

cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)

ret = close.pct_change()
fwd = ms.forward_ret(close, 10)

lib_panels = ms.library_panel(close, macro)

print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")

out = {"end": END, "horizon": 10, "results": {}}
print(f"{'factor':24s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'turn':>6s} {'maxrho':>7s}  GATE")
for name in sorted(lib_panels.keys()):
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
    }


def _med_div(df):
    """Element-wise: value / cross-sectional median for each date."""
    return df.div(df.median(axis=1), axis=0)


def cand_vol_adj_mom(close, win=20, skip=5, volwin=20):
    r = close.pct_change()
    mom = close / close.shift(win + skip) - 1.0
    rm = mom.subtract(mom.median(axis=1), axis=0)
    vol = r.rolling(volwin).std()
    return rm.multiply(_med_div(vol), axis=0)


def cand_ds_managed_mom(close, win=20, skip=5, volwin=20):
    r = close.pct_change()
    mom = close / close.shift(win + skip) - 1.0
    rm = mom.subtract(mom.median(axis=1), axis=0)
    neg = r.where(r < 0, 0.0)
    ds = (neg ** 2).rolling(volwin).mean().apply(np.sqrt)
    tot = r.rolling(volwin).std()
    ratio = ds / tot.add(1e-9)
    return rm.multiply(_med_div(ratio), axis=0)


def cand_breadth(close, win=20):
    r = close.pct_change()
    pos = (r.rolling(win).sum() > 0).astype(float)
    return pos.subtract(pos.mean(axis=1), axis=0)


def cand_skew(close, win=20):
    r = close.pct_change()
    return r.rolling(win, min_periods=14).skew()


cands = {
    "vol_adj_mom_20": cand_vol_adj_mom(close),
    "ds_managed_mom_20": cand_ds_managed_mom(close),
    "breadth_20": cand_breadth(close),
    "skew_20d": cand_skew(close),
}

print("\n--- Candidate exploration (horizon 10) ---")
for name, c in cands.items():
    ic = ms.daily_ic(c, fwd)
    st = ms.ic_stats(ic, 10)
    cov = ms.coverage_stats(c, fwd)
    turn = ms.rank_turnover(c, window=10)
    mrho, pairs = ms.max_lib_corr(c, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE) else "fail"
    print(f"{name:24s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    out["cand_" + name] = {
        "ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
        "coverage_dates_ge8": cov["coverage_dates_ge8"],
        "turnover": turn, "max_abs_library_correlation": round(mrho, 4),
        "corr_pairs": pairs,
    }

with open("scripts/miner3_20330216_revalidation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote scripts/miner3_20330216_revalidation.json")