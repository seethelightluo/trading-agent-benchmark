import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date'); x.date=pd.to_datetime(x.date); P[s]=x.set_index('date').close
p=pd.DataFrame(P).sort_index().ffill(); lr=np.log(p).diff()
# medium momentum normalized by downside risk, lagged one day
f=(np.log(p/p.shift(10))/lr.where(lr<0).rolling(40,min_periods=20).std()).shift(1).replace([np.inf,-np.inf],np.nan)
ics={}
for h in [1,5,10,20]:
 a=[];ns=[]
 for dt in f.index:
  y=np.log(p.shift(-h).loc[dt]/p.loc[dt]); z=pd.concat([f.loc[dt],y],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(q): a.append(q); ns.append(len(z))
 a=pd.Series(a); ics[h]=a
 print('horizon',h,'dates',len(a),'avgN',round(np.mean(ns),2),'coverage',round(np.mean(np.array(ns)/15),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 if h==10:
  for nm,s in [('first',a.iloc[:len(a)//3]),('middle',a.iloc[len(a)//3:2*len(a)//3]),('recent',a.iloc[2*len(a)//3:])]: print(nm,len(s),round(s.mean(),6),round(s.mean()/s.std(ddof=1),6))
rr=f.rank(axis=1,pct=True);t=[]
for i in range(1,len(rr)):
 z=pd.concat([rr.iloc[i-1],rr.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print('turnover',round(np.mean(t),6),'calendar_dates',len(f),'assets',len(p.columns))
f.index.name='date';f.reset_index().to_csv('scripts/miner_1_20320823_medium_downside_signal.csv',index=False)
