"""Probe data availability for the 15-asset tradable universe + macro signals."""
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

acct = get_account_dict()
print("watch_list:", acct.get("watch_list"))
print("total_assets:", acct.get("total_assets"))
print("available_cash:", acct.get("available_cash"))
print("positions:", acct.get("positions"))
print("orders:", len(acct.get("orders", [])))

wl = acct.get("watch_list", [])
print("\n--- stock data probe (days=800) ---")
for s in wl:
    try:
        df = get_stock_daily_data(symbol=s, days=800)
        if df is None:
            print(f"{s}: None")
        else:
            print(f"{s}: rows={len(df)} range={df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()} cols={list(df.columns)}")
    except Exception as e:
        print(f"{s}: ERROR {type(e).__name__}: {e}")

print("\n--- index data probe (days=800) ---")
for s in ["VIX", "DXY", "USDCNY", "USDJPY", "EURUSD"]:
    try:
        df = get_index_daily_data(symbol=s, days=800)
        if df is None:
            print(f"{s}: None")
        else:
            print(f"{s}: rows={len(df)} range={df['date'].iloc[0].date()}..{df['date'].iloc[-1].date()} cols={list(df.columns)}")
    except Exception as e:
        print(f"{s}: ERROR {type(e).__name__}: {e}")
