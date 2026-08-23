import json, numpy as np, pandas as pd

d = json.load(open('../persistent/date.json'))
valid_days = [x for x in d['trading_days'] if x <= d['visible_through']]
dates = pd.DatetimeIndex(valid_days)
print('num valid trading days through', d['visible_through'], len(dates))

universe = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
closes = {}
for s in universe:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')['close'].sort_index()
    closes[s] = df.reindex(dates).ffill()
px = pd.DataFrame(closes)
print('px shape', px.shape, 'last', px.index[-1].date())
print('NaN last row counts', px.isna().sum().to_dict())

for s in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    df = pd.read_csv(f'../persistent/index_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')['close'].sort_index()
    globals()[s] = df.reindex(dates).ffill()

for name, ser in [('VIX', VIX), ('DXY', DXY), ('USDCNY', USDCNY), ('USDJPY', USDJPY)]:
    cur = ser.iloc[-1]
    m20 = ser.iloc[-21] if len(ser) > 21 else ser.iloc[0]
    print(f'{name}: last={cur:.2f} 20d_ago={m20:.2f} chg={(cur/m20-1)*100:.1f}%')
print('recent 20d returns:')
print((px.iloc[-1]/px.iloc[-21]-1).sort_values().round(4).to_string())