"""miner_1 data availability check - verify API data window for factor research."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

WATCH = [
    "000300.SH", "SPX", "HSI", "N225", "SX5E", "000688.SH", "SOX", "NDX",
    "XAU", "COPPER", "WTI", "BTC", "ETH", "US10Y", "CN10Y",
]
MACRO = ["DXY", "USDCNY", "USDJPY", "EURUSD", "VIX"]

acct = get_account_dict()
print("watch_list from account:", acct.get("watch_list"))

for s in WATCH:
    df = get_stock_daily_data(s, days=1900)
    if df is None:
        print(f"{s}: NONE")
        continue
    print(f"{s}: rows={len(df)} range={df['date'].iloc[0]}..{df['date'].iloc[-1]} vol_nz={(df['volume']>0).sum()}")

for s in MACRO:
    df = get_index_daily_data(s, days=1900)
    if df is None:
        print(f"{s}: NONE (index)")
        continue
    print(f"{s}: rows={len(df)} range={df['date'].iloc[0]}..{df['date'].iloc[-1]}")
