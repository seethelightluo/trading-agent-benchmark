"""miner_3 revalidation through 2033-04-13 + candidate exploration."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2033-04-13"
H = 10

cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)
ret = close.pct_change()
fwd = ms.forward_ret(close, 10)
lib_panels = ms.library_panel(close, macro)

print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")

out = {"end": END, "horizon": H, "results": {}}
print(f"{'factor':24s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'covD8':>6s} {'turn':>6s} {'maxrho':>7s}  GATE")
for name in sorted(lib_panels.keys()):
    panel = lib_panels[name]
    ic = ms.daily_ic(panel, fwd)
    st = ms.ic_stats(ic, H)
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
        "corr_pairs": pairs, "gate": gate,
    }


def _med_div(df):
    return df.div(df.median(axis=1), axis=0)


def cand_reverse_mom20(close, win=20, skip=5, look=60):
    mom = close / close.shift(win + skip) - 1.0
    rm = mom.subtract(mom.median(axis=1), axis=0)
    lng = rm.rolling(look).mean()
    return -rm.multiply((lng < 0).astype(float), axis=0)


def cand_hl_loc(close, win=20):
    hi = close.rolling(win).max()
    lo = close.rolling(win).min()
    return (close - lo) / (hi - lo + 1e-9)


def cand_garch_ratio(close, win=20, bigwin=60):
    r = close.pct_change()
    s = r.rolling(win).std()
    l = r.rolling(bigwin).std()
    return -s.div(l.add(1e-9))


def cand_20d_cvar(close, win=20):
    r = close.pct_change()
    neg = r.where(r < 0, 0.0)
    q = r.rolling(win).quantile(0.1)
    return -q


cands = {
    "reverse_mom20_cond": cand_reverse_mom20(close),
    "hl_location_20": cand_hl_loc(close),
    "vol_stability_20x60": cand_garch_ratio(close),
    "cvar_20d": cand_20d_cvar(close),
    "skew_20d": close.pct_change().rolling(20, min_periods=14).skew(),
    "usd_impulse_beta_60x20": (lambda c, m: -c.pct_change().rolling(60, min_periods=30).cov(m.pct_change())/m.pct_change().rolling(60, min_periods=30).var() * (m.pct_change().rolling(20).mean()<0).astype(float))(close, macro["DXY"]),
}

print("\n--- Candidate exploration (horizon 10) ---")
for name, c in cands.items():
    ic = ms.daily_ic(c, fwd)
    st = ms.ic_stats(ic, H)
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
        "corr_pairs": pairs, "gate": gate,
    }

with open("scripts/miner3_20330413_revalidation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote scripts/miner3_20330413_revalidation.json")