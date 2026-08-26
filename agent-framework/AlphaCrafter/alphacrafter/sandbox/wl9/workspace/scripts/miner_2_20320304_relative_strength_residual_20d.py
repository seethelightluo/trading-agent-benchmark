import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for s in U:
 d=get_stock_daily_data(s,days=4500)
 if d is not None and len(d)>150: C[s]=d[['date','close']].dropna().drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(C).sort_index(); r=p.pct_change();
# Relative strength: asset's 20d return less contemporaneous cross-sectional median, lagged.
ret20=p.pct_change(20); csmed=ret20.median(axis=1); f=ret20.sub(csmed,axis=0).shift(1)
print('loaded_assets',len(C),'dates',len(p),'universe',len(U))
for h in [5,10,20,40,60]:
 fr=p.shift(-h)/p-1; qs=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); ds.append(dt)
 q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
 print('H',h,'dates',len(q),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/(q.std(ddof=1)+1e-12),(q>0).mean()))
# regime split
h=20; fr=p.shift(-h)/p-1; qs=[]; ds=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ds.append(dt)
q=pd.Series(qs,index=pd.to_datetime(ds)).dropna()
for label,a,b in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-28','2026','2028-12-31'),('2029-31','2029','2031-12-31'),('2032','2032','2032-12-31')]:
 z=q.loc[(q.index>=a)&(q.index<=b)]; print('REG',label,len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/(z.std(ddof=1)+1e-12)))
print('coverage %.6f turnover %.6f'%(f.notna().mean().mean(),f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20320304_relative_strength_residual_20d_signal.csv',index=False)
