from alphacrafter.sim.utils import get_stock_daily_data
for s in ['SPX','000300.SH','BTC','XAU','US10Y','VIX','EURUSD','DXY']:
    df = get_stock_daily_data(s, 10)
    if df is None:
        print(s, 'None')
    else:
        print(s, 'rows(10d):', len(df), 'last date:', df['date'].iloc[-1])
        print('  cols:', list(df.columns))
