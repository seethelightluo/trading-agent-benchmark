import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>150:
  d=d.copy(); d.date=pd.to_datetime(d.date); P[s]=d.sort_values('date').set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); px=px.loc[:'2030-06-12']
r=px.pct_change(); v20=r.rolling(20,min_periods=15).std(); v60=r.rolling(60,min_periods=40).std(); v120=r.rolling(120,min_periods=80).std()
# Volatility-compression breakout: medium momentum, penalized by current risk,
# and rewarded when short volatility is below its own medium baseline.
compression=(v60/v120).clip(.4,2.5)
f=px.pct_change(40).div(v60).mul(1/compression)
# cross-sectional robust demeaning, causal and interpretable
f=f.sub(f.median(axis=1),axis=0)
yields={}
for h in [5,10,20]:
 y=px.shift(-h)/px-1; ics=[]; ns=[]; ds=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=q.iloc[:,0].corr(q.iloc[:,1],method='spearman')
   if pd.notna(z): ics.append(z); ns.append(len(q)); ds.append(dt)
 ic=pd.Series(ics,index=ds)
 print('horizon',h,'dates',len(ic),'mean_n',round(np.mean(ns),2),'IC',round(ic.mean(),6),'ICIR',round(ic.mean()/ic.std(ddof=1),6),'hit',round((ic>0).mean(),4),'coverage',round(len(ic)/len(px),4))
 for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-06-12')]:
  q=ic.loc[(ic.index>=lo)&(ic.index<=hi)]
  if len(q): print('regime',lo,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
 if h==10: out=ic
# rank turnover on dates where signals available
ranks=f.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).dropna().mean()
print('turnover',round(turnover,6),'assets',len(P),'rows',len(px))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20300613_vol_compression_signal.csv',index=False)
