import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv'); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill(); r=px.pct_change()
# Multi-horizon trend agreement: medium momentum is trusted only when short and medium signs agree.
lag=r.shift(1); r5=lag.rolling(5,min_periods=5).sum(); r20=lag.rolling(20,min_periods=15).sum(); r60=lag.rolling(60,min_periods=40).sum()
agree=((np.sign(r5)==np.sign(r20)) & (np.sign(r20)==np.sign(r60))).astype(float)
raw=(r20+r60/3.0)*agree
f=raw.sub(raw.mean(axis=1),axis=0).replace([np.inf,-np.inf],np.nan)
print('instruments',len(U),'dates',len(px),'coverage',f.notna().mean().mean(),'agreement',agree.mean().mean())
for h in [1,5,10,20]:
 a=[];ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2:a.append(z.x.corr(z.y));ns.append(len(z))
 q=pd.Series(a).dropna(); print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n',len(q),'avgN',np.mean(ns))
a=[]
for i in range(len(px)-1):
 z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+1]/px.iloc[i]-1).rename('y')],axis=1).dropna()
 if len(z)>=8 and z.x.nunique()>2:a.append((px.index[i],z.x.corr(z.y)))
q=pd.Series(dict(a)); print('daily thirds',*[q.iloc[j*len(q)//3:(j+1)*len(q)//3].mean() for j in range(3)])
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
f.index=f.index.astype(str); f.to_csv('scripts/miner_3_20331017_agreement_momentum_signal.csv')
