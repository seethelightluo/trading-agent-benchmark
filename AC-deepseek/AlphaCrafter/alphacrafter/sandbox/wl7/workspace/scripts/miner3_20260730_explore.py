"""miner_3 data exploration: check availability for 15-asset universe + macro signals."""
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

print("=== TRADABLE ASSETS ===")
for s in WATCH:
    df = get_stock_daily_data(s, days=4000)
    if df is None or len(df) == 0:
        print(f"{s}: NO DATA")
        continue
    print(f"{s}: {len(df)} rows, {df['date'].min().date()} .. {df['date'].max().date()}, "
          f"vol_nan={df['volume'].isna().mean():.2%}")

print("\n=== MACRO SIGNALS ===")
for s in MACRO:
    try:
        df = get_index_daily_data(s, days=4000)
        if df is None or len(df) == 0:
            print(f"{s}: NO DATA")
            continue
        print(f"{s}: {len(df)} rows, {df['date'].min().date()} .. {df['date'].max().date()}")
    except Exception as e:
        print(f"{s}: ERROR {e}")

print("\n=== CROSS-ASSET RETURN CORRELATION (last 60d) ===")
closes = {}
for s in WATCH:
    df = get_stock_daily_data(s, days=4000)
    if df is not None and len(df) > 0:
        closes[s] = df.set_index("date")["close"].astype(float)
panel = pd.concat(closes, axis=1)
rets = panel.pct_change().tail(60)
print(rets.corr().round(2))
