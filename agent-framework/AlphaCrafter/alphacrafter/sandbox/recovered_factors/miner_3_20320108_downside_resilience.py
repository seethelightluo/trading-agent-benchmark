import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); m=r.median(axis=1)
# Downside resilience: asset excess return on negative-median-market sessions, normalized by own volatility.
down=(m<0); ex=r.sub(m,axis=0).where(down[:,None] if False else down, np.nan)
f=ex.rolling(60,min_periods=15).mean()/r.rolling(20,min_periods=15).std()
print('DATA',len(px),len(A),'coverage',round(f.notna().mean().mean(),4))
for h in [1,5,10,20]:
 y=px.shift(-h)/px-1; q=[]; ns=[]
 for d in px.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 q=np.array(q); print('H',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(np.nanmean(q),6),'ICIR',round(np.nanmean(q)/np.nanstd(q,ddof=1),6),'hit',round(np.mean(q>0),4))
print('turn10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),4))
for lo,hi in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2032')]:
 q=[]
 for d in px.loc[lo:hi].index:
  z=pd.concat([f.loc[d],(px.shift(-10)/px-1).loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('REG',lo,hi,len(q),round(np.nanmean(q),6),round(np.nanmean(q)/np.nanstd(q,ddof=1),6))
# proxy correlation evidence
for name,s in {'mom20':px.pct_change(20),'invvol':-r.rolling(20).std(),'downcap':r.where(m<0).rolling(60,min_periods=15).mean()/r.rolling(20).std()}.items():
 z=pd.concat([f.stack(),s.stack()],axis=1).dropna();print('CORR',name,round(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,6),len(z))
