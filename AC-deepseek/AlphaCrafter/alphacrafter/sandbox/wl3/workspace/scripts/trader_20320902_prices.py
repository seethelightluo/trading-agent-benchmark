from alphacrafter.sim.utils import get_stock_daily_data

for sym in ["SPX", "NDX", "SOX", "COPPER", "WTI", "XAU", "ETH", "000300.SH", "000688.SH"]:
    df = get_stock_daily_data(symbol=sym, days=16)
    if df is None:
        print(sym, "NO DATA")
        continue
    df = df.sort_values("date")
    last = df.iloc[-1]
    prev = df.iloc[-11]  # ~10 trading days ago (block start)
    print(
        f"{sym}: block_start_close={prev['close']:.2f} last_close={last['close']:.2f} "
        f"chg={(last['close'] / prev['close'] - 1) * 100:+.2f}% "
        f"last_date={last['date'].date()}"
    )
