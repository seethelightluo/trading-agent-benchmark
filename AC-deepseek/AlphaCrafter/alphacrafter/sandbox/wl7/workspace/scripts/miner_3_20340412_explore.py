"""miner_3 candidate factor exploration, visible end 2034-04-11.

Tests several novel candidate ideas vs h=10 forward returns in the current
VIX-elevated, alternating risk-off tape. Gates: |IC|>=0.0070, |ICIR|>=0.0840,
prefer max_abs_library_correlation < 0.5. No lookahead.
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2034-04-11"
cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)
ret = close.pct_change()
fwd = ms.forward_ret(close, 10)
lib_panels = ms.library_panel(close, macro)
print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")

cands = {}

# C1: cross-sectional rank momentum 30d demeaned (robust moment, skip5)
mom30 = close / close.shift(35) - 1.0
cands["rank_mom_30d_skip5"] = mom30.subtract(mom30.median(axis=1), axis=0)

# C2: mean-reversion z-score contrar (20d) - contrarian tilt
ma20 = close.rolling(20).mean()
v20 = ret.rolling(20).std()
z = ((close - ma20) / close) / v20
cands["z_ma20_contr"] = -z

# C3: downside-vol-adjusted momentum (momentum / downside vol) - risk-scaled trend
mom20 = close / close.shift(25) - 1.0
neg = ret.where(ret < 0, 0.0)
dsvol = (neg ** 2).rolling(20).mean().apply(np.sqrt)
cands["mom_div_downsidevol_20"] = mom20 / (dsvol + 1e-9)

# C4: defensive tilt: recent upside capture asymmetry (max vs |min| 20d) with contrarian lean
maxr = ret.rolling(20).max()
minr = ret.rolling(20).min()
cands["maxmin_20d_contr"] = -(maxr + minr)   # -range, mean-reversion

# C5: 10d momentum acceleration demeaned (accel of short trend vs longer)
m10 = close / close.shift(10) - 1.0
m60 = close / close.shift(60) - 1.0
accel = m10 - m60
cands["accel_10v60"] = accel.subtract(accel.median(axis=1), axis=0)

# C6: VIX-regime-gated downside-vol ratio (lean defensive in high vol)
dsr = -(dsvol / ret.rolling(20).std())
fwd_high = macro["VIX"] > macro["VIX"].rolling(60).median()
cands["dsr20_vix_gated"] = dsr.multiply((fwd_high * 1.0 + (~fwd_high) * 0.01), axis=0)


def max_lib_corr(cand, lib_panels):
    flat = cand.stack()
    best = 0.0; pairs = {}
    for name, p in lib_panels.items():
        pflat = p.reindex(cand.index).stack()
        df = pd.concat([flat.rename("f"), pflat.rename("p")], axis=1).dropna()
        if len(df) < 30:
            continue
        rho = float(df["f"].corr(df["p"]))
        pairs[name] = round(rho, 4)
        if abs(rho) > best:
            best = abs(rho)
    return best, pairs


print(f"{'candidate':26s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s} {'maxrho':>7s}  GATE")
results = {}
for name, panel in cands.items():
    ic = ms.daily_ic(panel, fwd)
    st = ms.ic_stats(ic, 10)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    mrho, pairs = max_lib_corr(panel, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE and mrho < 0.5) else "fail"
    print(f"{name:26s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    s = ic.dropna()
    for lab, nd in (("1y", 365), ("3m", 91)):
        rs = s[s.index >= s.index.max() - np.timedelta64(nd, "D")]
        if len(rs):
            m = rs.mean(); sd = rs.std(ddof=1)
            print(f"    {lab}: IC {m:+.4f} ICIR {m/sd if sd>0 else float('nan'):.3f} hit {(rs>0).mean():.2f} n {len(rs)}")
    results[name] = {"ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
                     "coverage_dates_ge8": cov["coverage_dates_ge8"],
                     "turnover_10d": turn, "max_abs_library_correlation": round(mrho, 4),
                     "corr_pairs": pairs,
                     "decay": {str(h): round(ms.ic_stats(ms.daily_ic(panel, ms.forward_ret(close, h)), h)["ic"], 4)
                               for h in (1, 2, 3, 5, 10, 20)}}

with open("scripts/miner3_20340412_explore.json", "w") as h:
    json.dump(results, h, indent=2)
print("\nWrote scripts/miner3_20340412_explore.json")