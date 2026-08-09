import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; d={}
for a in A:
 x=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date'); d[a]=x[['close','high','low']]
df=pd.concat(d,axis=1).sort_index(); c=df.xs('close',axis=1,level=1); hi=df.xs('high',axis=1,level=1); lo=df.xs('low',axis=1,level=1)
r=c.pct_change(); atr=((hi-lo)/c).rolling(20,min_periods=15).mean(); vol=r.rolling(20,min_periods=15).std()
# Fixed inverse construction: fade unusually efficient 10-session displacement, scaled by range and volatility.
f=-(c/c.shift(10)-1)/(atr+1e-8)/(vol+1e-8); f=f.sub(f.median(axis=1),axis=0).clip(-50,50)
print('rows',len(c),'assets',len(A),'range',c.index.min().date(),c.index.max().date())
for h in [1,5,10,20]:
 fw=c.shift(-h)/c-1; xs=[]; ns=[]
 for t in f.index:
  z=pd.concat([f.loc[t],fw.loc[t]],axis=1).dropna()
  if len(z)>=8: xs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=np.array(xs); print('h',h,'dates',len(x),'meanN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/(x.std(ddof=1)+1e-12),6),'hit',round((x>0).mean(),4))
turn=[]
for a,b in zip(f.index[:-1],f.index[1:]):
 z=pd.concat([f.loc[a].rank(pct=True),f.loc[b].rank(pct=True)],axis=1).dropna()
 if len(z)>=8: turn.append(np.mean(abs(z.iloc[:,0]-z.iloc[:,1])))
print('coverage',round(f.notna().sum(axis=1).mean()/15,6),'turnover',round(np.mean(turn),6),'valid_cells',int(f.notna().sum().sum()))
for start,end in [('2020-01-01','2026-07-15'),('2026-07-16','2030-12-31'),('2031-01-01','2034-02-15')]:
 q=[]; fw=c.shift(-10)/c-1
 for t in f.loc[start:end].index:
  z=pd.concat([f.loc[t],fw.loc[t]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print('regime',start,end,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/(q.std(ddof=1)+1e-12),6))
