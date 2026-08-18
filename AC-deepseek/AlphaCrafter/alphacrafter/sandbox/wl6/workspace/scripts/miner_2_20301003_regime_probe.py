"""miner_2 regime probe 2030-10-03 (data visible through 2030-10-02)."""
import numpy as np
import pandas as pd

VISIBLE = "2030-10-02"
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
OBS = ['DXY', 'USDCNY', 'USDJPY', 'EURUSD', 'VIX']


def load(sym, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)]
    return df.set_index("date").sort_index()


px = pd.DataFrame({s: load(s)["close"].astype(float) for s in TRADABLE})
ret = px.pct_change()
vix = load("VIX", INDEX_DIR)["close"].astype(float)
print("panel rows:", len(px), px.index.min().date(), "->", px.index.max().date())

for h in (10, 20, 60, 120):
    r = (px.iloc[-1] / px.iloc[-1 - h] - 1.0)
    print(f"\n{h}d return to {px.index[-1].date()}:")
    print(r.sort_values(ascending=False).round(3).to_string())

print("\nLast 20d VIX path:")
print(vix.tail(25).round(1).to_string())
print("VIX chg 10d:", round(float(vix.iloc[-1] / vix.iloc[-11] - 1), 3),
      "| 20d:", round(float(vix.iloc[-1] / vix.iloc[-21] - 1), 3),
      "| 60d:", round(float(vix.iloc[-1] / vix.iloc[-61] - 1), 3))

spx = px["SPX"]
print("\nSPX last close:", round(float(spx.iloc[-1]), 1),
      "| 5d:", round(float(spx.iloc[-1]/spx.iloc[-6]-1), 4),
      "| 10d:", round(float(spx.iloc[-1]/spx.iloc[-11]-1), 4))