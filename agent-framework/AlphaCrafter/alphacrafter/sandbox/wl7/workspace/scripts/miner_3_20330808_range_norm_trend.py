import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change()
# Lagged range-normalized medium trend: recent 10d return relative to prior 40d true range, with volatility penalty
lag=px.shift(1)
ret10=lag.pct_change(10)
rng=(lag.rolling(40).max()-lag.rolling(40).min())/lag.rolling(40).mean()
vol=r.shift(1).rolling(20).std()
f=(ret10/rng/vol).replace([np.inf,-np.inf],np.nan)
print('instruments',len(D),'dates',len(px),'coverage',f.notna().mean().mean())
for h in [1,5,10,20]:
 a=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:a.append(z.x.corr(z.y))
 q=pd.Series(a).dropna(); print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n',len(q),'avgN',len(z))
# thirds for H10
h=10;a=[]
for i in range(len(px)-h):
 z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8 and z.x.nunique()>2:a.append((px.index[i],z.x.corr(z.y)))
q=pd.Series(dict(a));print('thirds',*[q.iloc[j*len(q)//3:(j+1)*len(q)//3].mean() for j in range(3)])
f.index=f.index.astype(str);f.to_csv('scripts/miner_3_20330808_range_norm_trend_signal.csv')
