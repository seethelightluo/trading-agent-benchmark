"""miner_3 data exploration: check universe coverage, calendars, volume availability."""
import pandas as pd
import numpy as np

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

CUTOFF = pd.Timestamp("2026-07-15")  # visible through previous completed day

def load(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] <= CUTOFF].set_index("date").sort_index()
    return df

print("=== Asset coverage through", CUTOFF.date(), "===")
for a in WATCH:
    df = load(f"../persistent/stock_data/{a}.csv")
    print(f"{a:10s} rows={len(df):5d} {df.index[0].date()} -> {df.index[-1].date()} "
          f"vol_nz={(df['volume']>0).mean():.2f} close_nan={df['close'].isna().mean():.2f}")

print("\n=== Macro coverage ===")
for a in MACRO:
    df = load(f"../persistent/index_data/{a}.csv")
    print(f"{a:10s} rows={len(df):5d} {df.index[0].date()} -> {df.index[-1].date()} "
          f"close_nan={df['close'].isna().mean():.2f}")

# Common-date alignment of closes
closes = {a: load(f"../persistent/stock_data/{a}.csv")["close"] for a in WATCH}
panel = pd.concat(closes, axis=1, join="inner").dropna()
print("\n=== Common-date close panel ===")
print("shape:", panel.shape, "date range:", panel.index[0].date(), "->", panel.index[-1].date())

# Number of assets per common date
cnt = panel.notna().sum(axis=1)
print("dates with >=8 assets:", (cnt >= 8).sum(), "of", len(panel))

# Volume panel
vols = {a: load(f"../persistent/stock_data/{a}.csv")["volume"] for a in WATCH}
vpanel = pd.concat(vols, axis=1, join="inner")
print("\nVolume stats (common dates):")
print(vpanel.describe().T[["mean", "min"]].round(1).to_string())
