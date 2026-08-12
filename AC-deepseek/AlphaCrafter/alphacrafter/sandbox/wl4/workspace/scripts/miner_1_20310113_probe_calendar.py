"""MINER_1 2031-01-13: check Jan-2031 trading calendar for SPX (proxy) and others."""
import pandas as pd

df = pd.read_csv("../persistent/stock_data/SPX.csv", parse_dates=["date"])
jan = df[(df["date"] >= "2031-01-01") & (df["date"] <= "2031-01-15")]
print(jan[["date", "close"]].to_string(index=False))

# Also check crypto which trade daily
btc = pd.read_csv("../persistent/stock_data/BTC.csv", parse_dates=["date"])
btc_jan = btc[(btc["date"] >= "2031-01-01") & (btc["date"] <= "2031-01-15")]
print("\nBTC:")
print(btc_jan[["date", "close"]].to_string(index=False))
