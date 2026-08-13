"""miner_1 2034-07-28: build panel cache through current date (no future data).
Universe: 15 tradable cross-asset instruments + 5 macro observation series.
"""
import numpy as np
import pandas as pd

TRADABLE = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX",
            "NDX", "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]
END = "2034-07-28"

closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
for sym in TRADABLE:
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv", parse_dates=["date"])
    df = df.set_index("date").sort_index()
    df = df[df.index <= END]
    closes[sym] = df["close"].astype(float)
    opens[sym] = df["open"].astype(float)
    highs[sym] = df["high"].astype(float)
    lows[sym] = df["low"].astype(float)
    vols[sym] = df["volume"].astype(float)

close_px = pd.DataFrame(closes).sort_index()
open_px = pd.DataFrame(opens).sort_index()
high_px = pd.DataFrame(highs).sort_index()
low_px = pd.DataFrame(lows).sort_index()
vol_px = pd.DataFrame(vols).sort_index()
ret = close_px.pct_change()

macro = {}
for s in MACRO:
    m = pd.read_csv(f"../persistent/index_data/{s}.csv", parse_dates=["date"])
    m = m.set_index("date").sort_index()
    m = m[m.index <= END]
    macro[s] = m["close"].astype(float)
macro_px = pd.DataFrame(macro).sort_index()

panel = {
    "close": close_px, "open": open_px, "high": high_px, "low": low_px,
    "vol": vol_px, "ret": ret, "macro": macro_px,
}
with open("scripts/panel_cache_20340728.pkl", "wb") as f:
    pd.to_pickle(panel, f)

print("close shape:", close_px.shape, close_px.index.min().date(), "->", close_px.index.max().date())
print("macro shape:", macro_px.shape)
r = ret
print("\nzero-return (flat artifact) counts last 200d:")
print((r.tail(200).abs() < 1e-12).sum().sort_values(ascending=False).to_dict())
print("\nVIX last 5:", macro_px["VIX"].tail(5).round(2).to_dict())
print("\n20d mean daily ret @last:", round(float(r.tail(20).mean().mean()), 5))
print("20d ann vol mean @last:", round(float(r.tail(20).std().mean() * np.sqrt(252)), 2))
print("20d cross-sectional dispersion:", round(float(r.tail(20).std(axis=1).mean()), 5))
print("\n20d asset mean daily ret @last:")
print(r.tail(20).mean().sort_values(ascending=False).round(5).to_dict())
print("\nvolume coverage (>0):")
print((vol_px > 0).mean().round(3).to_dict())
