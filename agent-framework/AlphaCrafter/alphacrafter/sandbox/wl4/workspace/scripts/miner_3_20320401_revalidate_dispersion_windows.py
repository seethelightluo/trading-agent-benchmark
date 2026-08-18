import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-03-31'); D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'] for s in U}; p=pd.concat(D,axis=1).sort_index().loc[:cut]; lr=np.log(p).diff(); r=lr.rolling(10).sum(); v=lr.rolling(20).std()*np.sqrt(20); d=r.std(axis=1); f=(-r.sub(r.mean(axis=1),axis=0)).div(v.clip(.005,1)).clip(-5,5).where(d>d.rolling(60).median()).shift(1); R=np.log(p.shift(-10)/p); z=[]
for t in f.index:
 a=pd.concat([f.loc[t],R.loc[t]],axis=1).dropna()
 if len(a)>=8:z.append((t,a.iloc[:,0].corr(a.iloc[:,1])))
a=pd.Series(dict(z))
for days in [365,730,1095]:
 q=a[a.index>=cut-pd.Timedelta(days=days)]; print(days,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
print('all',len(a),a.mean(),a.mean()/a.std(ddof=1))
