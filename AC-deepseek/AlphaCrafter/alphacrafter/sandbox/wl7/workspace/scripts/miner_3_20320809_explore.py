"""miner_3 candidate factor exploration, visible end 2032-08-09.

Tests several candidate ideas against h=10 forward returns, reports IC/ICIR/hit,
coverage, turnover, and max abs library correlation vs the active library panel.
Gates: |IC|>=0.0070, |ICIR|>=0.0840. No lookahead.
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
ret = close.pct_change()

fwd = ms.forward_ret(close, 10)
lib_panels = ms.library_panel(close, macro)

# --- candidate factor constructions ---
cands = {}

# C1: vol-scaled momentum 60d. momentum normalized by trailing vol.
mom60 = close / close.shift(60) - 1.0
vol20 = ret.rolling(20).std()
cands["momsig_60_20"] = mom60 / vol20

# C2: 60d max drawdown (negative => drawdown). Avoid lookahead (running drawdown of realized past).
roll_max = close.rolling(60).max()
cands["maxdd_60"] = (close / roll_max - 1.0)  # positive when near high (low drawdown)

# C3: range position / efficiency: (close - min60)/(max60 - min60)
rmin = close.rolling(60).min()
rmax = close.rolling(60).max()
cands["range_pos_60"] = (close - rmin) / (rmax - rmin).replace(0, np.nan)

# C4: skewness of returns 60d
cands["skew_60"] = ret.rolling(60).skew()

# C5: downside/upside vol ratio variant: sharper (mean/vol) 60d risk-adjusted momentum
cands["sharpe_60"] = (close / close.shift(60) - 1.0) / (ret.rolling(60).std() * np.sqrt(60))

# C6: amihud illiquidity 20d (|ret| / volume) with volume from trading volume
# volume only available for stock data; use synthetic - skip if missing

# C7: 3m-1m basis - not available cleanly; skip

# C8: price distance from 200d MA (trend) normalized by vol
ma200 = close.rolling(200).mean()
cands["ma200_dist"] = (close / ma200 - 1.0)

# C9: relative 10d momentum to 60d (short vs medium - reversal/mean-reversion)
cands["rel_mom_10_60"] = (close / close.shift(15) - 1.0) - (close / close.shift(65) - 1.0)


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


print(f"{'candidate':18s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s} {'maxrho':>7s}  GATE")
results = {}
for name, panel in cands.items():
    ic = ms.daily_ic(panel, fwd)
    st = ms.ic_stats(ic, 10)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    mrho, pairs = max_lib_corr(panel, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE) else "fail"
    print(f"{name:18s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
    # recent 1y / 2y
    s = ic.dropna()
    r1 = s[s.index >= s.index.max() - np.timedelta64(365, "D")]
    r2 = s[s.index >= s.index.max() - np.timedelta64(730, "D")]
    for lab, rs in (("1y", r1), ("2y", r2)):
        if len(rs):
            m = rs.mean(); sd = rs.std(ddof=1)
            print(f"    {lab}: IC {m:+.4f} ICIR {m/sd if sd>0 else float('nan'):.3f} hit {(rs>0).mean():.2f} n {len(rs)}")
    results[name] = {"ic": st["ic"], "icir": st["icir"], "hit": st["hit"], "n": st["n"],
                     "coverage_dates_ge8": cov["coverage_dates_ge8"],
                     "turnover_10d": turn, "max_abs_library_correlation": round(mrho, 4),
                     "corr_pairs": pairs,
                     "decay": {str(h): round(ms.ic_stats(ms.daily_ic(panel, ms.forward_ret(close, h)), h)["ic"], 4)
                               for h in (1, 2, 3, 5, 10, 20)}}

with open("scripts/miner3_20320809_explore.json", "w") as h:
    json.dump(results, h, indent=2)
print("\nWrote scripts/miner3_20320809_explore.json")