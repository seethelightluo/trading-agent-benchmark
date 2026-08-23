import json, numpy as np, pandas as pd

d = json.load(open('../persistent/date.json'))
valid_days = [x for x in d['trading_days'] if x <= d['visible_through']]
dates = pd.DatetimeIndex(valid_days)
print('valid trading days through', d['visible_through'], len(dates))

universe = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
closes = {}
for s in universe:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')['close'].sort_index()
    closes[s] = df.reindex(dates).ffill()
px = pd.DataFrame(closes)

for s in ['VIX', 'DXY', 'USDCNY', 'USDJPY', 'EURUSD']:
    df = pd.read_csv(f'../persistent/index_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')['close'].sort_index()
    globals()[s] = df.reindex(dates).ffill()

np.save('px_frames.npy', px.values)
np.save('px_index.npy', dates.values.astype('datetime64[D]'))
pd.Series(universe).to_csv('px_cols.csv', index=False, header=False)
print('saved px frames', px.shape)