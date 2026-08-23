"""miner_3 candidate factor exploration, visible end 2032-09-05 (current cycle 2032-09-07).

Tests several novel candidate ideas vs h=10 forward returns. Reports IC/ICIR/hit,
coverage, turnover, decay, and max abs library correlation. No lookahead.
Gates: |IC|>=0.0070, |ICIR|>=0.0840. Prefer max_abs_library_correlation < 0.5.
"""
import json
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2032-09-05"
cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)
ret = close.pct_change()
fwd = ms.forward_ret(close, 10)
lib_panels = ms.library_panel(close, macro)

print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")

cands = {}

# C1: RSJ (Relative Strength + Jump): 20d momentum minus 20d max single-day ret
mom20 = close / close.shift(25) - 1.0
maxret = ret.rolling(20).max()
cands["rsj_20"] = mom20 - maxret

# C2: downside-capture semi-alpha relative to equal-weight market (60d)
def downside_capture(close, window=60):
    r = close.pct_change()
    mkt = r.mean(axis=1)
    mkt_neg = mkt.where(mkt < 0, np.nan)
    out = pd.DataFrame(index=r.index, columns=r.columns, dtype=float)
    for a in r.columns:
        cov = r[a].rolling(window).cov(mkt_neg)
        var = mkt_neg.rolling(window).var()
        out[a] = (cov / var) - 1.0
    return out
cands["downside_capture_60"] = downside_capture(close, 60)

# C3: vol-of-vol ratio (short / long vol)
v20 = ret.rolling(20).std()
v60 = ret.rolling(60).std()
cands["vol_ratio_20_60"] = v20 / v60

# C4: mean-reversion z-score from 20d mean (contrarian sign)
ma20 = close.rolling(20).mean()
z = (close - ma20) / (ret.rolling(20).std() * close)
cands["z_ma20_contr"] = -z

# C7: cross-asset relative momentum 120d demeaned (skip5)
mom120 = close / close.shift(125) - 1.0
cands["rel_mom_120d_skip5"] = mom120.subtract(mom120.median(axis=1), axis=0)

# C8: momentum acceleration (change in 20d momentum over 10d)
m20 = close / close.shift(20) - 1.0
m20_prev = close.shift(10) / close.shift(30) - 1.0
cands["mom_accel_20"] = m20 - m20_prev

# C9: VIX-gated momentum (leans long only when VIX elevated -> risk-adjusted)
vix = macro["VIX"]
vixz = (vix - vix.rolling(60).mean()) / vix.rolling(60).std()
mom60 = close / close.shift(60) - 1.0
cands["mom60_vix_high"] = mom60.multiply((vixz > 0.5).astype(float), axis=0)


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


print(f"{'candidate':24s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'cov':>5s} {'turn':>6s} {'maxrho':>7s}  GATE")
results = {}
for name, panel in cands.items():
    ic = ms.daily_ic(panel, fwd)
    st = ms.ic_stats(ic, 10)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    mrho, pairs = max_lib_corr(panel, lib_panels)
    gate = "PASS" if (abs(st["ic"]) >= ms.IC_GATE and abs(st["icir"]) >= ms.ICIR_GATE and mrho < 0.5) else "fail"
    print(f"{name:24s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.2f} {st['n']:5d} "
          f"{cov['coverage_dates_ge8']:5.2f} {turn:6.2f} {mrho:7.2f}  {gate}")
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

with open("scripts/miner3_20320907_explore.json", "w") as h:
    json.dump(results, h, indent=2)
print("\nWrote scripts/miner3_20320907_explore.json")