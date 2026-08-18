from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
import pandas as pd

for sym, fn in [("SPX", get_index_daily_data), ("SPX", get_stock_daily_data),
                ("XAU", get_stock_daily_data), ("BTC", get_stock_daily_data),
                ("VIX", get_index_daily_data)]:
    try:
        df = fn(sym, days=10)
        print(sym, fn.__name__, "None" if df is None else f"rows={len(df)} cols={list(df.columns)}")
        if df is not None and len(df):
            print(df.tail(3).to_string())
    except Exception as e:
        print(sym, fn.__name__, "ERR", repr(e)[:120])
    print("---")
