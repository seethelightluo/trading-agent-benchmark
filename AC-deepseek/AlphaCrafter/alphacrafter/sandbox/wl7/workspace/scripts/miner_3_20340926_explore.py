"""miner_3 candidate factor exploration, visible end 2034-09-26.

Tests novel candidate ideas vs h=10 forward returns in the current
alternating / VIX-flippy cross-asset tape. Gates: |IC|>=0.0070,
|ICIR|>=0.0840, prefer max_abs_library_correlation < 0.5. No lookahead.
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2034-09-26"
cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)
ret = close.pct_change()
fwd = ms.forward_ret(close, 10)
lib_panels = ms.library_panel(close, macro)
print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")

cands = {}

# C1: cross-sectional rank momentum 40d demeaned (slower trend, skip5)
mom40 = close / close.shift(45) - 1.0
cands["rank_mom_40d_skip5"] = mom40.subtract(mom40.median(axis=1), axis=0)

# C2: yield-curve steepening tilt (CN10Y-US10Y spread momentum 20d, directional cross-asset)
spread = close["CN10Y"] - close["US10Y"]
spread_mom = spread.shift(5) / spread.shift(25) - 1.0
def_pen = pd.Series([0.0]*15, index=close.columns)
for col in ["XAU", "US10Y", "CN10Y", "000688.SH", "000300.SH"]:
    def_pen[col] = 1.0
for col in ["BTC", "ETH", "N225", "SX5E", "WTI", "COPPER", "HSI", "SPX", "SOX", "NDX"]:
    def_pen[col] = -1.0
cands["yc_spread_mom_20d"] = pd.DataFrame(
    np.outer(spread_mom.values, def_pen.values), index=spread_mom.index, columns=close.columns)

# C4: max-drawdown persistence (20d drawdown magnitude, demeaned)
dd = close / close.rolling(20).max() - 1.0
cands["drawdown_20d_demean"] = dd.subtract(dd.median(axis=1), axis=0)

# C5: vol-stabilized momentum 20d (mom / realized vol), cross-demeaned
mom20 = close / close.shift(25) - 1.0
rv20 = ret.rolling(20).std()
vol_mom = mom20 / (rv20 + 1e-9)
cands["vol_scaled_mom_20d"] = vol_mom.subtract(vol_mom.median(axis=1), axis=0)

# C6: skewness tilt (negative skew mean-reversion) 20d skip5
rsk = ret.shift(5)
skew = rsk.rolling(20, min_periods=12).skew()
cands["skew_20d_skip5"] = skew

# C7: VIX-10d-spike defensive tilt (high VIX momentum => favor defensive assets)
vix_r = macro["VIX"] / macro["VIX"].shift(10) - 1.0
def_pen2 = pd.Series([0.0]*15, index=close.columns)
for col in ["XAU", "US10Y", "CN10Y", "000688.SH", "000300.SH"]:
    def_pen2[col] = 1.0
for col in ["BTC", "ETH", "N225", "SX5E", "WTI", "COPPER", "HSI"]:
    def_pen2[col] = -1.0
cands["vix10_def_tilt"] = pd.DataFrame(
    np.outer(vix_r.values, def_pen2.values), index=vix_r.index, columns=close.columns)

# C8: cross-sectional 60d total-rel-momentum (long-lag value anchor) demeaned
mom60 = close / close.shift(65) - 1.0
cands["rank_mom_60d_skip5"] = mom60.subtract(mom60.median(axis=1), axis=0)


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

with open("scripts/miner3_20340926_explore.json", "w") as h:
    json.dump(results, h, indent=2)
print("\nWrote scripts/miner3_20340926_explore.json")