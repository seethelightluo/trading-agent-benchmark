"""Check full data depth for all watchlist symbols via API."""
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ["000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
         "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y"]

for s in WATCH:
    df = None
    for fn in (get_stock_daily_data, get_index_daily_data):
        try:
            df = fn(s, days=3000)
            if df is not None and len(df):
                break
        except Exception:
            df = None
    if df is None or not len(df):
        print(f"{s:10s} NO DATA")
        continue
    print(f"{s:10s} rows={len(df):4d} first={df.iloc[0]['date']:%Y-%m-%d} last={df.iloc[-1]['date']:%Y-%m-%d} vol_nan={df['volume'].isna().sum()}")

print("\n--- volume sample check (SPX last 5 rows) ---")
df = get_stock_daily_data("SPX", days=5)
print(df[['date','close','volume','pct_change']].to_string(index=False))
