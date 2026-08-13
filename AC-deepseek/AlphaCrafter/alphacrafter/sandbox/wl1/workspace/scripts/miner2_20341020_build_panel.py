"""miner2 2034-10-20: rebuild aligned 15-name cross-asset panel strictly through visible_through=2034-10-19."""
import pandas as pd
import numpy as np

ASOF = "2034-10-19"
ASSETS = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX", "XAU",
          "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]


def load(sym):
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv", parse_dates=["date"])
    df = df[df["date"] <= ASOF].reset_index(drop=True)
    df = df.set_index("date").sort_index()
    return df


closes, opens, highs, lows, vols = {}, {}, {}, {}, {}
for a in ASSETS:
    d = load(a)
    closes[a] = d["close"]
    opens[a] = d["open"]
    highs[a] = d["high"]
    lows[a] = d["low"]
    vols[a] = d["volume"]

px = pd.DataFrame(closes).sort_index()
op = pd.DataFrame(opens).sort_index()
hi = pd.DataFrame(highs).sort_index()
lo = pd.DataFrame(lows).sort_index()
vo = pd.DataFrame(vols).sort_index()

panel = {"close": px, "open": op, "high": hi, "low": lo, "volume": vo}
with open("scripts/panel_cache_20341020.pkl", "wb") as f:
    pd.to_pickle(panel, f)

print("Panel saved scripts/panel_cache_20341020.pkl")
print("Date range:", px.index.min().date(), "->", px.index.max().date(), "rows:", len(px))
print("Last 3 rows per asset (close):")
print(px.tail(3).round(4).to_string())
na = px.isna().sum()
print("\nNaNs per asset (full):")
print(na[na > 0].to_string() if (na > 0).any() else "none")
print("\nLast date NaN count:", int(px.iloc[-1].isna().sum()))
