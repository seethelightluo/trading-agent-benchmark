from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

for a in ["000300.SH", "000688.SH", "CN10Y"]:
    df = get_stock_daily_data(a, days=70)
    if df is None or len(df) == 0:
        df = get_index_daily_data(a, days=70)
    if df is None:
        print(a, "-> None")
        continue
    print(a, "rows:", len(df), "first:", df.iloc[0]["date"], "last:", df.iloc[-1]["date"])
    print("  first close:", df.iloc[0]["close"], "last close:", df.iloc[-1]["close"])
    print("  last 5 closes:", list(df["close"].tail(5)))
