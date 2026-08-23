import json, numpy as np, pandas as pd

d = json.load(open('../persistent/date.json'))
valid_days = [x for x in d['trading_days'] if x <= d['visible_through']]

universe = ['000300.SH', 'SPX', 'HSI', 'N225', 'SX5E', '000688.SH', 'SOX', 'NDX',
            'XAU', 'COPPER', 'WTI', 'BTC', 'ETH', 'US10Y', 'CN10Y']
for s in universe:
    df = pd.read_csv(f'../persistent/stock_data/{s}.csv')
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values('date')
    print(f'{s:10s} rows={len(df):5d} first={df["date"].iloc[0].date()} last={df["date"].iloc[-1].date()}')