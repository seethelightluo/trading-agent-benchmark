from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCH = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

for s in WATCH:
    df = get_stock_daily_data(symbol=s, days=15)
    if df is None or len(df) < 12:
        df = get_index_daily_data(symbol=s, days=15)
    if df is None or len(df) < 12:
        print(s, "NO DATA")
        continue
    df = df.sort_values('date')
    p0 = df.iloc[-11]['close']   # close 10 trading days ago (block start region)
    p1 = df.iloc[-1]['close']
    r = (p1/p0 - 1)*100
    # also last 2 days for context
    r2 = (df.iloc[-1]['close']/df.iloc[-3]['close'] - 1)*100
    print(f"{s:10s} p_start={p0:12.4f} p_end={p1:12.4f} block10d={r:+7.2f}% last2d={r2:+6.2f}%  d0={df.iloc[-1]['date'].date()}")
