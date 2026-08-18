"""miner_2 regime probe 2032-08-19 (data visible through 2032-08-18)."""
import numpy as np
import pandas as pd

VISIBLE = "2032-08-18"
DATA_DIR = "../persistent/stock_data"
INDEX_DIR = "../persistent/index_data"
TRADABLE = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']


def load(sym, ddir=DATA_DIR):
    df = pd.read_csv(f"{ddir}/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= pd.Timestamp(VISIBLE)]
    return df.set_index("date").sort_index()


px = pd.DataFrame({s: load(s)["close"].astype(float) for s in TRADABLE})
ret = px.pct_change()
vix = load("VIX", INDEX_DIR)["close"].astype(float)
dxy = load("DXY", INDEX_DIR)["close"].astype(float)
usdcny = load("USDCNY", INDEX_DIR)["close"].astype(float)
print("panel rows:", len(px), px.index.min().date(), "->", px.index.max().date())

print("\n=== 5/10/20/60/120d returns to", px.index[-1].date(), "===")
for h in (5, 10, 20, 60, 120):
    r = (px.iloc[-1] / px.iloc[-1 - h] - 1.0)
    print(f"\n{h}d:")
    print(r.sort_values(ascending=False).round(3).to_string())

print("\n=== VIX / DXY / USDCNY last 30d ===")
for name, s in [("VIX", vix), ("DXY", dxy), ("USDCNY", usdcny)]:
    print(name, "last:", round(float(s.iloc[-1]), 2),
          "| 5d:", round(float(s.iloc[-1] / s.iloc[-6] - 1), 3),
          "| 20d:", round(float(s.iloc[-1] / s.iloc[-21] - 1), 3),
          "| 60d:", round(float(s.iloc[-1] / s.iloc[-61] - 1), 3))

spx = px["SPX"]
ma20 = spx.rolling(20).mean().iloc[-1]
ma60 = spx.rolling(60).mean().iloc[-1]
ma120 = spx.rolling(120).mean().iloc[-1]
print("\nSPX close:", round(float(spx.iloc[-1]), 1),
      "| vs MA20/60/120:", round(spx.iloc[-1]/ma20-1, 4),
      round(spx.iloc[-1]/ma60-1, 4), round(spx.iloc[-1]/ma120-1, 4))

disp20 = ret.tail(20).std(axis=1).mean()
disp60 = ret.tail(60).std(axis=1).mean()
print("avg daily cross-sectional dispersion 20d:", round(disp20, 4),
      "| 60d:", round(disp60, 4))

spx_vol20 = spx.pct_change().tail(20).std() * np.sqrt(252)
spx_vol60 = spx.pct_change().tail(60).std() * np.sqrt(252)
print("SPX ann. vol 20d:", round(spx_vol20, 3), "| 60d:", round(spx_vol60, 3))

print("\n=== 1y (250d) trailing returns ===")
print((px.iloc[-1] / px.iloc[-1 - 250] - 1.0).sort_values(ascending=False).round(2).to_string())

print("\n=== frozen check (recent non-NaN close count) ===")
for s in ['000300.SH', '000688.SH', 'HSI', 'CN10Y']:
    ids = px[s].dropna().index
    print(s, "last:", ids[-1].date() if len(ids) else None, "| n:", len(ids))