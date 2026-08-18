import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 try: d=get_index_daily_data(s,days=4000)
 except FileNotFoundError: d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d): D[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
px=pd.DataFrame(D).sort_index(); r20=px.pct_change(20); fac=r20.sub(r20.mean(axis=1),axis=0)
for h in [1,5,10,20]:
 vals=[];ns=[]; fr=px.shift(-h)/px-1
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z))
 a=np.array(vals);print(h,'dates',len(a),'meanN',round(np.mean(ns),2),'IC',round(np.nanmean(a),6),'ICIR',round(np.nanmean(a)/np.nanstd(a,ddof=1),6),'hit',round(np.mean(a>0),4))
r=fac.rank(axis=1,pct=True);print('turnover',round(r.diff().abs().mean(axis=1).dropna().mean(),6),'coverage',round(fac.notna().mean().mean(),4),'dates',len(px),'assets',len(D))
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2028')]:
 a=[];fr=px.shift(-10)/px-1
 for dt in fac.loc[lo:hi].index:
  z=pd.concat([fac.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(lo,hi,len(a),round(np.mean(a),6),round(np.mean(a)/np.std(a,ddof=1),6))
out=fac.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20280229_relative_momentum_signal.csv',index=False)
