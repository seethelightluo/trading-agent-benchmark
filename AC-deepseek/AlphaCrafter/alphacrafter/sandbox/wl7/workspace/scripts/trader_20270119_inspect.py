from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

for sym in ["SPX", "000300.SH"]:
    try:
        df = get_stock_daily_data(sym, days=5)
        print(sym, "stock daily:", type(df), None if df is None else (df.shape, list(df.columns)))
        if df is not None:
            print(df.tail(3))
    except Exception as e:
        print(sym, "stock err:", e)
    try:
        df2 = get_index_daily_data(sym, days=5)
        print(sym, "index daily:", type(df2), None if df2 is None else (df2.shape, list(df2.columns)))
        if df2 is not None:
            print(df2.tail(3))
    except Exception as e:
        print(sym, "index err:", e)

try:
    df3 = get_index_daily_data("VIX", days=5)
    print("VIX index daily:", None if df3 is None else (df3.shape, list(df3.columns)))
    if df3 is not None:
        print(df3.tail(3))
except Exception as e:
    print("VIX err:", e)
