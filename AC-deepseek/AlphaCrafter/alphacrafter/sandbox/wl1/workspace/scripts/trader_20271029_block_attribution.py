"""Trader block attribution: 2027-10-29 -> 2027-11-12."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data

W = {
    "000300.SH": 0.046, "SPX": 0.027, "HSI": 0.066, "N225": 0.076,
    "SX5E": 0.086, "000688.SH": 0.049, "SOX": 0.074, "NDX": 0.058,
    "XAU": 0.059, "COPPER": 0.107, "WTI": 0.076, "BTC": 0.100,
    "ETH": 0.066, "US10Y": 0.072, "CN10Y": 0.038,
}
print(f"sum weights: {sum(W.values()):.4f}")

rows = []
for a, w in W.items():
    df = get_stock_daily_data(symbol=a, days=40)
    if df is None or len(df) < 30:
        print(a, "NO DATA")
        continue
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date").sort_index()
    c = df["close"].astype(float)
    # block return: close at rebalance day (10-29) -> last close (11-12)
    if pd.Timestamp("2027-10-29") in c.index and len(c) >= 2:
        r_blk = c.iloc[-1] / c.loc[pd.Timestamp("2027-10-29")] - 1.0
    else:
        r_blk = np.nan
    # regime stats as of decision (data through 10-28/10-29)
    ret20 = c.pct_change().tail(20).mean()
    ma20 = c.rolling(20).mean().iloc[-1]
    below_ma = c.iloc[-1] < ma20
    mom120 = c.iloc[-6] / c.iloc[-126] - 1.0 if len(c) > 126 else np.nan
    rows.append((a, w, r_blk, ret20, below_ma, mom120))
    print(f"{a:10s} w={w*100:5.1f}%  blk_ret={r_blk*100:6.2f}%  "
          f"20dmean={ret20*100:6.2f}%  belowMA20={below_ma}  mom120={mom120*100:6.1f}%")

tot = sum(w * r for _, w, r, *_ in rows if np.isfinite(r))
print(f"\nweighted block PnL (approx): {tot*100:.2f}%")

# VIX regime at decision
try:
    vix = pd.read_csv("../persistent/index_data/VIX.csv")
    vix["date"] = pd.to_datetime(vix["date"])
    vix = vix[vix["date"] <= pd.Timestamp("2027-10-29")].sort_values("date")
    v = vix.set_index("date")["close"].astype(float)
    print(f"\nVIX at 10-29: {v.iloc[-1]:.2f}  (20d ago: {v.iloc[-21]:.2f}, "
          f"chg: {(v.iloc[-1]/v.iloc[-21]-1)*100:.1f}%)")
except Exception as e:
    print("vix err", e)
