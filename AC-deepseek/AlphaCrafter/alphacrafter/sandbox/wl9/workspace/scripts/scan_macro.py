import pandas as pd, numpy as np
vis=pd.Timestamp('2034-10-25')
for a in ['VIX','DXY']:
    df=pd.read_csv(f'../persistent/index_data/{a}.csv')
    s=pd.Series(pd.to_numeric(df['close'],errors='coerce').values,
                index=pd.to_datetime(df['date'])).sort_index()
    s=s[s.index<=vis].dropna()
    close=s.iloc[-1]
    r10=close/s.iloc[-11]-1
    idx30=[x for x in s.index if x>=vis-pd.Timedelta(days=30)]
    r30=close/s.loc[idx30].iloc[0]-1
    r120=close/s.iloc[-121]-1
    hi=s.max()
    print(f'{a}: close={close:.2f} r10={r10*100:.1f}% r30={r30*100:.1f}% r120={r120*100:.1f}% max={hi:.2f}')