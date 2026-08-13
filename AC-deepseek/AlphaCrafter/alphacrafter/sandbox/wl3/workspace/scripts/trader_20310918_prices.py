from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

syms = ["000300.SH", "SPX", "N225", "000688.SH", "SOX", "NDX", "XAU", "COPPER", "WTI", "ETH", "BTC", "HSI", "SX5E", "US10Y", "CN10Y"]
for s in syms:
    try:
        df = get_stock_daily_data(symbol=s, days=30)
    except Exception:
        df = get_index_daily_data(symbol=s, days=30)
    if df is None or len(df) == 0:
        print(s, "NO DATA")
        continue
    df = df.sort_values("date")
    tail = df.tail(6)
    for _, r in tail.iterrows():
        print(f"{s}: {str(r['date'])[:10]} close={r['close']:.4f}")
    print("---")
