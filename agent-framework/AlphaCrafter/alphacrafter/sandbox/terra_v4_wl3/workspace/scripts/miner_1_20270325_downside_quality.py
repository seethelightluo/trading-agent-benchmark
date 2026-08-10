import pandas as pd,numpy as np,glob,os
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={}
for a in watch:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index(); C[a]=d.close[d.index<=cut]
close=pd.DataFrame(C).sort_index(); r=close.pct_change()
# Defensive quality: inverse downside volatility, mildly rewarded by stable positive drift.
down=r.where(r<0,0).rolling(40,min_periods=20).std()
trend=r.rolling(20,min_periods=15).sum()
fac=(-down/(r.rolling(40,min_periods=20).std()+1e-8) + 0.20*trend/(r.rolling(40,min_periods=20).std()+1e-8))
fac.to_csv('scripts/miner_1_20270325_downside_quality_signal.csv')
print('assets',len(watch),'rows',len(fac),'period',fac.index.min(),fac.index.max())
for h in [1,5,10]:
 y=close.pct_change(h).shift(-h); vals=[]; ds=[]; ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds); print('H',h,'dates',len(s),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,hi,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage %.4f turnover %.4f'%(fac.notna().sum(axis=1).mean()/len(watch),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
