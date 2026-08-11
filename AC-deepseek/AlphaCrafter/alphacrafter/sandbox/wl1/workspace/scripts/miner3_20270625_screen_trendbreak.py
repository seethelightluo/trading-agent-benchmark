"""miner_3: explore trend-break / drawdown family (one idea: short-term trend breakdown
of extended names). Motivation from screener feedback: momentum top-picks (WTI->NDX->SOX)
keep crashing post-rebalance; a guard that de-ranks names that broke their short-term MA
after a run-up should reduce whipsaw. Panel up to 2027-06-24.
"""
import numpy as np
import pandas as pd
import pickle

panel = pickle.load(open("scripts/panel_cache.pkl", "rb"))
C = panel["close"]
lr = C.pct_change()

gate_ic, gate_icir = 0.0070, 0.0840
mom120 = C.shift(5) / C.shift(125) - 1.0
ma20 = C.rolling(20).mean()
ma60 = C.rolling(60).mean()
rmax20 = C.rolling(20).max()
rmax60 = C.rolling(60).max()
rmin20 = C.rolling(20).min()

factors = {}
# trend distance: positive above MA, negative below
factors["ma20_dist"] = C / ma20 - 1.0
factors["ma60_dist"] = C / ma60 - 1.0
# drawdown from recent high: negative in drawdown
factors["ddraw_20"] = C / rmax20 - 1.0
factors["ddraw_60"] = C / rmax60 - 1.0
# range position (0=at 20d low, 1=at 20d high)
factors["rngpos_20"] = (C - rmin20) / (rmax20 - rmin20)
# interaction: extended momentum x trend break (de-ranking guard)
factors["ext_break20"] = mom120.rank(axis=1) * (C / ma20 - 1.0)
# trend-break flag: -1 if broke below MA20 after being above recently
below20 = (C < ma20).astype(float)
was_above = (C.shift(5) > ma20.shift(5)).astype(float)
factors["brk_20d"] = -(below20 * was_above)  # -1 on fresh break, 0 otherwise

def daily_ic_series(f, h):
    fwd = C.shift(-h) / C - 1.0
    ics = []
    for dt in f.index:
        ff, rr = f.loc[dt], fwd.loc[dt]
        m = ff.notna() & rr.notna()
        if m.sum() < 8:
            continue
        ic = ff[m].rank().corr(rr[m].rank())
        if np.isfinite(ic):
            ics.append((dt, ic))
    return pd.Series({d: v for d, v in ics})

print(f"{'factor':14s} {'h':>3s} {'IC':>9s} {'ICIR':>9s} {'hit':>6s} {'n':>5s} {'IC12m':>9s} {'ICIR12m':>9s}  gate")
for name, f in factors.items():
    best = None
    for h in (1, 2, 3, 5, 10):
        s = daily_ic_series(f, h)
        if len(s) == 0:
            continue
        ic = s.mean(); sd = s.std(ddof=1)
        icir = ic / sd if sd > 0 else 0
        if best is None or abs(icir) > abs(best[1]):
            hit = float((s > 0).mean()) if ic > 0 else float((s < 0).mean())
            best = (h, icir, ic, len(s), hit, s)
    if best is None:
        print(f"{name:14s}  no data"); continue
    h, icir, ic, n, hit, s = best
    cut = s.index.max() - pd.Timedelta(days=365)
    s12 = s[s.index >= cut]
    ic12 = s12.mean(); icir12 = ic12 / s12.std(ddof=1) if s12.std(ddof=1) > 0 else 0
    ok = "PASS" if abs(ic) >= gate_ic and abs(icir) >= gate_icir else "fail"
    print(f"{name:14s} {h:3d} {ic:+9.5f} {icir:+9.5f} {hit:6.3f} {n:5d} {ic12:+9.5f} {icir12:+9.5f}  {ok}")
