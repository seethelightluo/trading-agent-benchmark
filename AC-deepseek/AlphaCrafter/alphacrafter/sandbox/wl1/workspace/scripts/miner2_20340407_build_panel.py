"""miner2 2034-04-07: build fresh daily panel (tradable + macro) through 2034-04-06."""
import pandas as pd, numpy as np, pickle, os

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
END = "2034-04-06"

closes, vols, highs, lows, opens = {}, {}, {}, {}, {}
for s in WATCH:
    df = pd.read_csv(f"../persistent/stock_data/{s}.csv", parse_dates=["date"])
    df = df[df["date"] <= END].set_index("date").sort_index()
    closes[s] = df["close"].astype(float)
    vols[s] = df["volume"].astype(float)
    highs[s] = df["high"].astype(float)
    lows[s] = df["low"].astype(float)
    opens[s] = df["open"].astype(float)

close_px = pd.DataFrame(closes).sort_index()
vol_px = pd.DataFrame(vols).sort_index()
high_px = pd.DataFrame(highs).sort_index()
low_px = pd.DataFrame(lows).sort_index()
open_px = pd.DataFrame(opens).sort_index()
ret = close_px.pct_change()

macro = {}
for s in MACRO:
    df = pd.read_csv(f"../persistent/index_data/{s}.csv", parse_dates=["date"])
    df = df[df["date"] <= END].set_index("date").sort_index()
    macro[s] = df["close"].astype(float)
macro_px = pd.DataFrame(macro).sort_index()

# fwd returns on aligned close index (for horizon h, fwd[h] = close.shift(-h)/close - 1)
fwd = {}
for h in (1, 2, 3, 5, 10, 20, 30):
    fwd[h] = close_px.shift(-h) / close_px - 1.0

panel = {"close": close_px, "open": open_px, "high": high_px, "low": low_px,
         "vol": vol_px, "ret": ret, "macro": macro_px, "fwd": fwd}
with open("scripts/panel_cache_20340406.pkl", "wb") as fh:
    pickle.dump(panel, fh, protocol=4)

print("panel:", close_px.shape, close_px.index.min().date(), "->", close_px.index.max().date())
print("cols:", list(close_px.columns))
print("macro last:", {k: str(macro_px[k].dropna().index.max().date()) for k in macro_px})
print("n_valid_per_asset:", close_px.notna().sum().to_dict())
print("last rows close:")
print(close_px.tail(3))
