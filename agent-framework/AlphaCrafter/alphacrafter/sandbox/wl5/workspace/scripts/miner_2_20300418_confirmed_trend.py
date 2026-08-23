import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>150:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); r20=px.pct_change(20); r5=px.pct_change(5); vol=r.rolling(30).std()*np.sqrt(252)
# trend persistence: medium-term trend, rewarded only when short-term direction confirms; relative to cross-sectional median
rel20=r20.sub(r20.median(axis=1),axis=0); rel5=r5.sub(r5.median(axis=1),axis=0)
f=rel20/vol * np.sign(rel5)
f=f.sub(f.mean(axis=1),axis=0)
for h in [5,10,20]:
 y=px.shift(-h)/px-1; vals=[];ds=[];ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q):vals.append(q);ds.append(dt);ns.append(len(z))
 ic=pd.Series(vals,index=ds);print('H',h,'dates',len(ic),'mean_n',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(px),4))
 for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-04-17')]:
  q=ic.loc[(ic.index>=a)&(ic.index<=b)]
  if len(q):print(' ',a,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
print('assets',len(P),'rows',len(px),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'asset_coverage',round(f.notna().mean().mean(),6))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300418_confirmed_trend_signal.csv',index=False)
