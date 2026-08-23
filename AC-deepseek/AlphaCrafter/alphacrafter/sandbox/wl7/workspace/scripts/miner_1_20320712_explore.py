import os, pandas as pd, numpy as np

BASE = '../persistent'
def load_close():
    watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
    idx={}
    for s in watch:
        p=os.path.join(BASE,'stock_data',s+'.csv')
        if not os.path.exists(p): continue
        df=pd.read_csv(p)
        df['date']=pd.to_datetime(df['date'])
        df=df.set_index('date').sort_index()
        idx[s]=df['close']
    X=pd.DataFrame(idx)
    return X

X=load_close()
print("shape", X.shape)
print("cols", list(X.columns))
print("nan per col", X.isna().sum().to_dict())
print("date range", X.index.min(), X.index.max())
print(X.tail(3))