from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

syms = ["000300.SH", "SPX", "N225", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "ETH", "BTC", "HSI", "SX5E", "US10Y", "CN10Y"]
target_dates = {"2031-08-19", "2031-08-20", "2031-08-21", "2031-09-02", "2031-09-03", "2031-09-04", "2031-09-05", "2031-09-08"}
for s in syms:
    try:
        df = get_stock_daily_data(symbol=s, days=45)
    except Exception:
        df = get_index_daily_data(symbol=s, days=45)
    if df is None or len(df) == 0:
        print(s, "NO DATA")
        continue
    df = df.sort_values("date")
    print(s, end=": ")
    for _, r in df.iterrows():
        d = str(r["date"])[:10]
        if d in target_dates:
            print(f"{d}={r['close']:.4f}", end=" ")
    print()
