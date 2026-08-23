"""Check data availability as of current sim date for revalidation."""
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data

WATCHLIST = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']

for sym in WATCHLIST:
    df = get_stock_daily_data(symbol=sym, days=2600)
    if df is not None:
        print(sym, len(df), df['date'].iloc[0], df['date'].iloc[-1])
    else:
        print(sym, 'None')

for sym in ['VIX','DXY','USDJPY','USDCNY','EURUSD']:
    df = get_index_daily_data(symbol=sym, days=2600)
    if df is not None:
        print(sym, len(df), df['date'].iloc[0], df['date'].iloc[-1])
    else:
        print(sym, 'None')