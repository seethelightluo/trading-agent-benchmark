import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>150:
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill().loc[:'2030-06-12']; r=px.pct_change()
# Downside-adjusted intermediate momentum: reward 20d return, penalize only
# recent negative-return variability; designed as a defensive complement.
down=r.where(r<0,0).rolling(40,min_periods=25).std()
raw=px.pct_change(20).div(down.replace(0,np.nan)); sig=raw.sub(raw.median(axis=1),axis=0)
for h in [5,10,20]:
 y=px.shift(-h)/px-1; vals=[]; ns=[]; ds=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): vals.append(q);ns.append(len(z));ds.append(dt)
 a=pd.Series(vals,index=ds); print('horizon',h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4),'coverage',round(len(a)/len(px),4))
 for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-06-12')]:
  q=a.loc[lo:hi]
  if len(q): print('regime',lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
 if h==10: out=a
print('turnover',round(sig.rank(axis=1,pct=True).diff().abs().mean().mean(),6),'assets',len(P),'rows',len(px))
sig.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300613_downside_adj_signal.csv',index=False)
