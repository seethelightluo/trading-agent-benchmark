"""miner_3 revalidation + new candidate exploration, visible end 2032-12-27."""
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
    }


# ---- New candidates ----
# 1) XAU/real-asset tilt: rank of asset's 20d return vs cross-sectional median,
#    combined with low inflation-hedge (commodity) volatility — expresses rotation
# 2) skew_20d: momentum of downside skew (captures tail risk posture)
def cand_skew(close, window=20, min_periods=14):
    ret = close.pct_change()
    return ret.rolling(window, min_periods=min_periods).skew()


def cand_vol_of_vol(close, window=20):
    r = close.pct_change()
    vol = r.rolling(window).std()
    return vol / vol.shift(window).rolling(window).mean().add(1e-9) - 1.0

def cand_rsi_14(close, window=14):
    r = close.pct_change()
    gain = r.clip(lower=0).rolling(window).mean()
    loss = (-r.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.add(1e-9)
    return 100 - 100 / (1 + rs)

def cand_downside_dev(close, window=20):
    r = close.pct_change()
    neg = r.where(r < 0, 0.0)
    return ((neg**2).rolling(window).mean().apply(np.sqrt))

# trend strength combined with absence of big drawdown
def cand_trend_less_dd(close, window=40):
    mom = close / close.shift(window) - 1.0
    maxv = close.rolling(window).max()
    dd = close / maxv - 1.0
    return mom + dd  # high momentum low drawdown

cands = {
    "skew_20d": cand_skew(close),
    "vol_of_vol_20d": cand_vol_of_vol(close),
    "rsi_14": cand_rsi_14(close),
    "downside_dev_20d": cand_downside_dev(close),
    "trend_less_dd_40": cand_trend_less_dd(close),
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
    out["results"][name] = {
        "ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
        "coverage_dates_ge8": cov["coverage_dates_ge8"],
        "turnover": turn, "max_abs_library_correlation": round(mrho, 4),
        "corr_pairs": pairs,
    }

with open("scripts/miner3_20321227_revalidation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote scripts/miner3_20321227_revalidation.json")