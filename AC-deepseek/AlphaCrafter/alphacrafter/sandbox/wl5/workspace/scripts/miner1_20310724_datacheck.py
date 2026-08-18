"""miner_1: inspect data characteristics for factor design (2031-07-24 cycle)."""
import pandas as pd, numpy as np

VISIBLE = "2031-07-23"
WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def load_asset(sym):
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= pd.Timestamp(VISIBLE)].sort_values("date").reset_index(drop=True)

# 1) yield series levels
for s in ["US10Y", "CN10Y"]:
    df = load_asset(s)
    print(s, "last close:", df["close"].iloc[-1], "first:", df["close"].iloc[0],
          "pct_change stats: mean", round(df["pct_change"].mean(), 6), "std", round(df["pct_change"].std(), 6),
          "min", round(df["pct_change"].min(), 4), "max", round(df["pct_change"].max(), 4))

# 2) volume availability across assets
print("\nVolume stats (nonzero share, last 500 rows):")
for s in WATCH:
    df = load_asset(s)
    v = df["volume"].tail(500)
    print(f"{s:10s} nonzero={100*(v>0).mean():.1f}% mean={v.mean():.3g}")

# 3) correlation of returns across assets (sample of the cross-section)
closes = {}
for s in WATCH:
    df = load_asset(s)
    closes[s] = df.set_index("date")["close"]
px = pd.DataFrame(closes).sort_index()
r = px.pct_change().tail(500)
print("\nReturn correlation (last 500d), SPX vs others:")
print(r.corr().loc["SPX"].round(2).to_string())

# 4) US10Y series check: is it a yield (small values) or price index?
print("\nUS10Y last 5 rows:", load_asset("US10Y")[["date", "open", "close", "high", "low"]].tail(5).to_string())
print("CN10Y last 5 rows:", load_asset("CN10Y")[["date", "open", "close", "high", "low"]].tail(5).to_string())
