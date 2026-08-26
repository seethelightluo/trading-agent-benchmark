import pandas as pd
from pathlib import Path
SD=Path('../persistent/stock_data'); ID=Path('../persistent/index_data')
ASSETS=['000300.SH','000688.SH','SPX','HSI','N225','SX5E','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
for a in ASSETS:
    f=SD/f'{a}.csv'
    if not f.exists(): f=ID/f'{a}.csv'
    d=pd.read_csv(f,parse_dates=['date'])
    print(a, f.parent.name, d['date'].min().date(), d['date'].max().date(), len(d), d.columns.tolist())
print('---macro---')
for m in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
    d=pd.read_csv(ID/f'{m}.csv',parse_dates=['date'])
    print(m, d['date'].min().date(), d['date'].max().date(), len(d))