# -*- coding: utf-8 -*-
"""miner_1: inspect data characteristics for factor design (2031-08-07 cycle, visible through 2031-08-06)."""
import pandas as pd, numpy as np

VISIBLE = "2031-08-06"
WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

def load_asset(sym):
    df = pd.read_csv(f"../persistent/stock_data/{sym}.csv")
    df["date"] = pd.to_datetime(df["date"])
    return df[df["date"] <= pd.Timestamp(VISIBLE)].sort_values("date").reset_index(drop=True)

print("=== data ranges ===")
for s in WATCH:
    df = load_asset(s)
    print(f"{s:10s} rows={len(df):5d} {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")

# 1) yield series levels
print("\n=== yield series ===")
for s in ["US10Y", "CN10Y"]:
    df = load_asset(s)
    print(s, "last close:", df["close"].iloc[-1], "first:", df["close"].iloc[0],
          "pct_change stats: mean", round(df["pct_change"].mean(), 6), "std", round(df["pct_change"].std(), 6),
          "min", round(df["pct_change"].min(), 4), "max", round(df["pct_change"].max(), 4))

# 2) volume availability across assets
print("\n=== volume stats (nonzero share, last 500 rows) ===")
for s in WATCH:
    df = load_asset(s)
    v = df["volume"].tail(500)
    print(f"{s:10s} nonzero={100*(v>0).mean():.1f}% mean={v.mean():.3g}")

# 3) correlation of returns across assets (sample)
closes = {}
for s in WATCH:
    df = load_asset(s)
    closes[s] = df.set_index("date")["close"]
px = pd.DataFrame(closes).sort_index()
r = px.pct_change().tail(500)
print("\n=== return correlation (last 500d), SPX vs others ===")
print(r.corr().loc["SPX"].round(2).to_string())

# 4) regime snapshot last 120 days
print("\n=== regime snapshot (last 120d returns) ===")
ret120 = px.pct_change(120).iloc[-1].sort_values()
print(ret120.round(3).to_string())

# 5) cross-sectional return dispersion (10d)
disp10 = px.pct_change(10).tail(300).std(axis=1)
print("\n10d cross-sectional dispersion: mean", round(disp10.mean(), 4), "last", round(disp10.iloc[-1], 4))
