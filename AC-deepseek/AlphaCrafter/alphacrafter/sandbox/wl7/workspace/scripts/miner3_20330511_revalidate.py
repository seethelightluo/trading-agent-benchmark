"""miner_3 revalidation of active library + candidate exploration through 2033-05-10."""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2033-05-10"
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
    s = ic.dropna()
    row = {"ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
           "coverage_dates_ge8": cov["coverage_dates_ge8"], "turnover": turn,
           "max_abs_library_correlation": round(mrho, 4), "gate": gate}
    for lab, nd in (("1y", 365), ("6m", 183), ("3m", 91)):
        ss = s[s.index >= s.index.max() - np.timedelta64(nd, "D")]
        if len(ss) == 0:
            continue
        m = ss.mean(); sd = ss.std(ddof=1)
        ir = m / sd if sd > 0 else np.nan
        print(f"    {lab}: IC {m:+.4f} ICIR {ir:+.3f} hit {(ss>0).mean():.2f} n {len(ss)}")
        row[f"recent_{lab}"] = {"ic": float(m), "icir": float(ir), "hit": float((ss > 0).mean()), "n": int(len(ss))}
    out["results"][name] = row

# Candidate exploration
def cand_reverse_mom20(close, win=20, skip=5, look=60):
    mom = close / close.shift(win + skip) - 1.0
    rm = mom.subtract(mom.median(axis=1), axis=0)
    lng = rm.rolling(look).mean()
    return -rm.multiply((lng < 0).astype(float), axis=0)

def cand_hl_location(close, win=20):
    hi = close.rolling(win).max(); lo = close.rolling(win).min()
    return (close - lo) / (hi - lo + 1e-9)

def cand_vol_stability(close, win=20, bigwin=60):
    r = close.pct_change()
    return -r.rolling(win).std().div(r.rolling(bigwin).std().add(1e-9))

def cand_cvar(close, win=20):
    return -close.pct_change().rolling(win).quantile(0.1)

def cand_zscore_price(close, win=20):
    return (close - close.rolling(win).mean()) / close.rolling(win).std()

def cand_macd_hist(close, fast=12, slow=26, sig=9):
    ema_f = close.ewm(span=fast, adjust=False).mean()
    ema_s = close.ewm(span=slow, adjust=False).mean()
    macd = ema_f - ema_s
    return macd - macd.ewm(span=sig, adjust=False).mean()

def cand_skew_ewma(close, win=20, skip=3):
    r = close.pct_change()
    return r.shift(skip).rolling(win, min_periods=12).skew()

cands = {
    "reverse_mom20_cond": cand_reverse_mom20(close),
    "hl_location_20": cand_hl_location(close),
    "vol_stability_20x60": cand_vol_stability(close),
    "cvar_20d": cand_cvar(close),
    "zscore_price_20": cand_zscore_price(close),
    "macd_hist": cand_macd_hist(close),
    "skew_20d_skip3": cand_skew_ewma(close),
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
    out["cand_" + name] = {"ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
                           "coverage_dates_ge8": cov["coverage_dates_ge8"], "turnover": turn,
                           "max_abs_library_correlation": round(mrho, 4), "gate": gate}

with open("scripts/miner3_20330511_revalidation.json", "w") as f:
    json.dump(out, f, indent=2)
print("\nWrote scripts/miner3_20330511_revalidation.json")