from alphacrafter.sim.utils import get_stock_daily_data

for a in ["NDX", "BTC", "ETH", "000300.SH", "SOX", "SPX"]:
    f = get_stock_daily_data(a, days=25)
    f = f.sort_values("date")
    print(f"\n=== {a} ===")
    for _, row in f.tail(14).iterrows():
        print(f"{str(row['date'])[:10]}  close={row['close']:.2f}")
