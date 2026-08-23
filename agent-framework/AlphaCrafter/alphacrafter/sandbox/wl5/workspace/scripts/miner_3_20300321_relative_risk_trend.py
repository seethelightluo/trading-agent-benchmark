import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
fr={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date); fr[s]=x.drop_duplicates('date').set_index('date').close
p=pd.concat(fr,axis=1).sort_index(); r=p.pct_change()
# Relative trend: asset 20d return relative to contemporaneous cross-sectional median,
# attenuated when its own 20d volatility is extreme (causal).
m=p.pct_change(20); cs=m.sub(m.median(axis=1),axis=0)
v=r.rolling(20).std(); f=cs/(v*np.sqrt(252))
# modestly favor relative winners only when broad median trend is positive; otherwise inverse signal
# primary candidate is relative risk-normalized trend (direction tested below)
for h in [5,10,20]:
 fw=p.shift(-h)/p-1; vals=[]; ds=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ds.append(dt); ns.append(len(z))
 q=pd.Series(vals,index=ds).dropna(); print('h',h,'dates',len(q),'meanN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 print('inverse IC',round(-q.mean(),6),'inverse ICIR',round(-q.mean()/q.std(ddof=1),6))
print('coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(pct=True).diff().abs().mean().mean(),6),'assets',len(fr),'range',p.index.min(),p.index.max())
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-03-20')]:
 fw=p.shift(-10)/p-1; a=[]
 for dt in p.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(a).dropna();print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20300321_relative_risk_trend_signal.csv',index=False);print('artifact rows',len(out))
