from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data

assets = get_account_dict()["watch_list"]
print("watch_list:", assets)
print()
for a in assets:
    df = None
    try:
        df = get_stock_daily_data(a, days=15)
    except Exception:
        df = None
    if df is None or len(df) < 12:
        try:
            df = get_index_daily_data(a, days=15)
        except Exception:
            df = None
    if df is None or len(df) < 12:
        print(f"{a}: insufficient data")
        continue
    df = df.sort_values("date")
    last = df.iloc[-1]
    p0 = df.iloc[-11]["close"]
    p1 = df.iloc[-1]["close"]
    print(f"{a}: last_date={last['date'].date()} ret_10d={p1/p0-1:.4%} close={p1:.4f}")
