import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; z={}
for s in U:
 try:d=get_index_daily_data(s,days=4000)
 except:d=None
 if d is None:
  try:d=get_stock_daily_data(s,days=4000)
  except:d=None
 if d is not None:z[s]=d.set_index(pd.to_datetime(d.date)).close.astype(float)
p=pd.DataFrame(z).sort_index().ffill(); r=p.pct_change(); vol=r.rolling(20).std(); disp=vol.mean(axis=1)
# short reversal activated when cross-asset dispersion is high; signal lagged
f=-r.rolling(5).sum().mul(disp.gt(disp.rolling(120).quantile(.65)),axis=0)/vol
rows=[]
for i in range(len(p)-10):
 q=pd.concat([f.iloc[i],p.iloc[i+10]/p.iloc[i]-1],axis=1).dropna()
 if len(q)>=8: rows.append((p.index[i],len(q),q.iloc[:,0].corr(q.iloc[:,1],method='spearman')))
x=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date'); a=x.ic.dropna()
print('assets',len(z),'dates',len(a),'mean_n',x.n.mean(),'coverage',len(a)/(len(p)-10))
print('IC',a.mean(),'std',a.std(),'ICIR',a.mean()/a.std(),'hit',(a>0).mean(),'r180',a.tail(180).mean(),'r500',a.tail(500).mean())
for h in [1,5,10,20]:
 q=[]
 for i in range(len(p)-h):
  t=pd.concat([f.iloc[i],p.iloc[i+h]/p.iloc[i]-1],axis=1).dropna()
  if len(t)>=8:q.append(t.iloc[:,0].corr(t.iloc[:,1],method='spearman'))
 q=pd.Series(q).dropna();print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(),'dates',len(q))
print('turnover',f.rank(pct=True).diff().abs().mean().mean(),'active',disp.gt(disp.rolling(120).quantile(.65)).mean())
