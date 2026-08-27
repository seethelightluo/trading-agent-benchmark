import pandas as pd, numpy as np

assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX',
        'XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
vis=pd.Timestamp('2034-10-25')
rows=[]
for a in assets:
    df=pd.read_csv(f'../persistent/stock_data/{a}.csv')
    s=pd.Series(pd.to_numeric(df['close'],errors='coerce').values,
                index=pd.to_datetime(df['date'])).sort_index()
    s=s[s.index<=vis].dropna()
    if len(s)<130:
        rows.append((a,'short',len(s))); continue
    close=s.iloc[-1]
    r10=close/s.iloc[-11]-1
    idx30=[x for x in s.index if x>=vis-pd.Timedelta(days=30)]
    r30=close/s.loc[idx30].iloc[0]-1
    r120=close/s.iloc[-121]-1
    rows.append((a,round(close,2),round(r10*100,1),round(r30*100,1),round(r120*100,1)))
print('asset          close     r10%   r30%   r120%')
for r in rows:
    if len(r)==5:
        print(f'{r[0]:12s} {r[1]:10.2f} {r[2]:6.1f} {r[3]:6.1f} {r[4]:6.1f}')
    else:
        print(r)