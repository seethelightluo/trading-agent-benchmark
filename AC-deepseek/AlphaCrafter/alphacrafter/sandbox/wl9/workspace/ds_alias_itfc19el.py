import pandas as pd
import json
d=json.load(open('../persistent/date.json'))
# probe volume availability for all assets
for a in ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']:
    df=pd.read_csv(f'../persistent/stock_data/{a}.csv')
    v=pd.to_numeric(df['volume'],errors='coerce')
    print(a, 'rows',len(df),'volume nonnull', v.notna().sum(), 'vol>0', (v>0).sum(), 'first vol>0 idx', v[v>0].index[0] if (v>0).any() else None)