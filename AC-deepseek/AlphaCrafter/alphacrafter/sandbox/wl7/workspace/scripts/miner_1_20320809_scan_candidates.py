"""miner_1 exploratory scan 2032-08-09 (visible through 2032-08-06).

Scan several novel, interpretable factor constructions on the 15-asset
cross-asset universe. This is exploration only: report IC/ICIR/hit per horizon,
per-year stability, coverage, turnover. Dedicated validation script follows for
any candidate passing the admission gates (|IC|>=0.0070, |ICIR|>=0.0840, h=10).
"""
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
import miner_shared as ms

END = "2032-08-06"

cal = ms.master_calendar(END)
close = ms.load_close(END)
macro = ms.load_macro(END)
ret = close.pct_change()

print(f"Panel through {END}: {len(cal)} dates, {close.shape[1]} assets")


def ic_row(panel, h=10):
    fwd = ms.forward_ret(close, h)
    ic = ms.daily_ic(panel, fwd)
    st = ms.ic_stats(ic, h)
    cov = ms.coverage_stats(panel, fwd)
    turn = ms.rank_turnover(panel, window=10)
    return st, cov, turn


def ic_by_year(panel, h=10):
    fwd = ms.forward_ret(close, h)
    ic = ms.daily_ic(panel, fwd).dropna()
    df = pd.DataFrame({"ic": ic})
    df["year"] = ic.index.year
    out = []
    for y, g in df.groupby("year"):
        m = g["ic"].mean()
        sd = g["ic"].std(ddof=1)
        out.append((y, round(float(m), 4), round(float(m / sd), 3) if sd > 0 else np.nan, int(len(g))))
    return out


def decay(panel):
    return {str(h): round(ms.ic_stats(ms.daily_ic(panel, ms.forward_ret(close, h)), h)["ic"], 4)
            for h in (1, 2, 3, 5, 10, 20)}


# ---------------- Candidate definitions ----------------
candidates = {}

# 1. Skewness of daily returns over 20d, skip 5d (tail-shape asymmetry)
candidates["skew_20d_skip5"] = ret.shift(5).rolling(20, min_periods=12).skew()

# 2. Risk-adjusted return: 60d mean daily ret / std (quality/Sharpe style)
candidates["sharpe_60d"] = (ret.rolling(60, min_periods=30).mean()
                            / ret.rolling(60, min_periods=30).std())

# 3. Drawdown depth: distance from 60d rolling high (negative)
candidates["drawdown_60d"] = close / close.rolling(60, min_periods=30).max() - 1.0

# 4. Short/long vol ratio: 10d vol / 60d vol (vol-regime tilt)
vol10 = ret.rolling(10, min_periods=6).std()
vol60 = ret.rolling(60, min_periods=30).std()
candidates["vol_ratio_10x60"] = vol10 / vol60

# 5. Up-day streak density: fraction of positive days over 20d, skip 5d
pos = (ret > 0).astype(float)
candidates["up_streak_20d_skip5"] = pos.shift(5).rolling(20, min_periods=12).mean()

# 6. BTC-lead momentum: BTC 10d return as cross-asset signal for all assets
btc_ret10 = ret["BTC"].rolling(10).mean()
candidates["btc_lead_10d"] = btc_ret10.to_frame("BTC").reindex(columns=close.columns).ffill().bfill()

print(f"\n{'candidate':20s} {'IC':>8s} {'ICIR':>7s} {'hit':>5s} {'n':>5s} {'covAD':>6s} {'cov8':>6s} {'turn':>6s}")
for name, panel in candidates.items():
    st, cov, turn = ic_row(panel)
    print(f"{name:20s} {st['ic']:8.4f} {st['icir']:7.3f} {st['hit']:5.3f} {st['n']:5d} "
          f"{cov['coverage_asset_days']:6.3f} {cov['coverage_dates_ge8']:6.3f} {turn:6.2f}")

print("\nDecay (IC by horizon):")
for name, panel in candidates.items():
    print(f"  {name:20s} {decay(panel)}")

print("\nPer-year IC (h=10):")
for name, panel in candidates.items():
    print(f"  {name:20s} {ic_by_year(panel)}")
