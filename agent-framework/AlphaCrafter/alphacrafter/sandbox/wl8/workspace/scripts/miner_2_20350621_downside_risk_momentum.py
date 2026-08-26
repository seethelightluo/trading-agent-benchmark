import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fs={}
for s in U:
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<80:d=get_index_daily_data(s,5000)
 if d is not None: fs[s]=d.set_index(pd.to_datetime(d.date)).close
px=pd.DataFrame(fs).sort_index().ffill(); r=px.pct_change()
# Candidate: lagged 30-session return scaled by downside deviation, rewarding persistent gains while penalizing losses.
down=r.where(r<0,0).rolling(30).std(); sig=(px.pct_change(30)/down.replace(0,np.nan)).shift(1)
ics=[]; ns=[]; trs=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],(px.shift(-10)/px-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: ics.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman')));ns.append(len(z))
 a=sig.loc[dt].rank(pct=True); b=sig.shift(1).loc[dt].rank(pct=True); q=pd.concat([a,b],axis=1).dropna()
 if len(q):trs.append(np.mean(abs(q.iloc[:,0]-q.iloc[:,1])))
ic=pd.Series(dict(ics)).dropna(); print('dates',len(ic),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',ic.mean(),'ICIR',ic.mean()/ic.std()*np.sqrt(252),'hit',np.mean(ic>0),'turnover',np.mean(trs))
for h in [1,5,10,20]:
 v=[]; y=px.shift(-h)/px-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:v.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(v),len(v))
for n in [365,750,1260]:
 q=ic.tail(n);print('recent',n,q.mean(),q.mean()/q.std()*np.sqrt(252),len(q))
print('range',ic.index.min(),ic.index.max())
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_2_20350621_downside_risk_momentum_signal.csv',index=False)
pd.DataFrame({'date':ic.index,'ic':ic.values}).to_csv('scripts/miner_2_20350621_downside_risk_momentum_ic.csv',index=False)
