"""Quick environment / data coverage check for miner_1 on 2034-06-22."""
from alphacrafter.sim.utils import (
    get_stock_daily_data,
    get_index_daily_data,
    get_account_dict,
)

acc = get_account_dict()
print('ACCOUNT watch_list:', acc.get('watch_list'))
print('ACCOUNT net_assets:', acc.get('net_assets'))

WL = ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for s in WL:
    df = get_index_daily_data(symbol=s, days=10)
    if df is None or len(df) == 0:
        df2 = get_stock_daily_data(symbol=s, days=10)
        print(s, 'via stock:', None if df2 is None else (len(df2), df2['date'].iloc[0], df2['date'].iloc[-1]))
    else:
        print(s, 'via index:', len(df), df['date'].iloc[0], df['date'].iloc[-1], float(df['close'].iloc[-1]))

for s in ['DXY','VIX','USDCNY','USDJPY','EURUSD']:
    df = get_index_daily_data(symbol=s, days=5)
    print('macro', s, None if df is None else (len(df), df['date'].iloc[-1], float(df['close'].iloc[-1])))