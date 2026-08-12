import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x)>150:
  x=x[['date','close']]; x.date=pd.to_datetime(x.date); D[s]=x.drop_duplicates('date').set_index('date').close
p=pd.DataFrame(D).sort_index().ffill(); r=np.log(p).diff()
# Recovery reversal: favor assets with strong long trend but recent underperformance,
# scaled by downside risk and gated by positive trend.
long=np.log(p/p.shift(60)); pull=np.log(p/p.shift(10)); dn=r.where(r<0,0).rolling(40).std()
trend=(long>0).astype(float); f=(-(pull-0.20*long)/(dn+1e-8))*trend
f=f.replace([np.inf,-np.inf],np.nan).shift(1)
for h in [1,5,10,20]:
 v=[]
 for i in range(len(p)-h):
  z=pd.concat([f.iloc[i],np.log(p.iloc[i+h]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(v).dropna(); print('h',h,'obs',len(q),'ic',q.mean(),'icir',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
print('dates',len(p),'assets',len(D),'coverage',f.notna().sum(axis=1).mean()/len(U),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [(0,len(p)//3),(len(p)//3,2*len(p)//3),(2*len(p)//3,len(p))]:
 v=[]
 for i in range(a,min(b,len(p)-20)):
  z=pd.concat([f.iloc[i],np.log(p.iloc[i+20]/p.iloc[i])],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(v).dropna(); print('regime',p.index[a].date(),p.index[min(b-1,len(p)-1)].date(),len(q),q.mean(),q.mean()/q.std(ddof=1))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20311030_recovery_reversal_signal.csv',index=False)
