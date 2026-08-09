"""Data exploration: check availability/history for the 15-instrument universe."""
import pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data, get_account_dict

acct = get_account_dict()
watch = acct.get("watch_list", [])
print("WATCH_LIST:", watch)
print("ACCT KEYS:", list(acct.keys()))
print("TOTAL ASSETS:", acct.get("total_assets"), "CASH:", acct.get("available_cash"))

print("\n=== INSTRUMENT DATA AVAILABILITY ===")
for sym in watch:
    df = None
    try:
        df = get_stock_daily_data(symbol=sym, days=4000)
    except Exception as e1:
        try:
            df = get_index_daily_data(symbol=sym, days=4000)
        except Exception as e2:
            print(sym, "ERR", e1, e2)
            continue
    if df is None or len(df) == 0:
        print(f"{sym}: NO DATA")
        continue
    print(f"{sym}: rows={len(df)} start={df['date'].iloc[0].date()} end={df['date'].iloc[-1].date()} cols={list(df.columns)}")
