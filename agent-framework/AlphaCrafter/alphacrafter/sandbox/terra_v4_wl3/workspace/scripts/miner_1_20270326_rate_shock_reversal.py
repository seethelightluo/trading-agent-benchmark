import pandas as pd,numpy as np
from scipy.stats import spearmanr
cut=pd.Timestamp('2027-03-24'); watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
C={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().close for a in watch}
close=pd.DataFrame(C).sort_index(); r=close.pct_change(); market=r.median(axis=1).fillna(r.mean(axis=1))
spread=np.log(close['US10Y'].clip(lower=1e-9))-np.log(close['CN10Y'].clip(lower=1e-9)); shock=spread.diff(5); scale=shock.abs().rolling(120,min_periods=20).mean(); intensity=(shock.abs()/(scale+1e-8)).clip(0,4).fillna(0)
F={}
for a in watch:
 beta=r[a].rolling(60,min_periods=20).cov(market)/market.rolling(60,min_periods=20).var(); resid=r[a]-beta*market
 F[a]=(-resid.rolling(3,min_periods=3).sum()*(1+0.25*intensity)).clip(-.2,.2)
fac=pd.DataFrame(F); fac.to_csv('scripts/miner_1_20270326_rate_shock_reversal_signal.csv')
print('assets',len(watch),'rows',len(fac))
for h in [1,5,10]:
 y=close.pct_change(h).shift(-h); vals=[];ds=[];ns=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ds.append(dt);ns.append(len(z))
 s=pd.Series(vals,index=ds);print('H',h,'dates',len(s),'avgN %.2f IC %.6f ICIR %.6f hit %.4f'%(np.mean(ns),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
 if h==1:
  for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2027-03-24')]:
   q=s[(s.index>=lo)&(s.index<=hi)];print('regime',lo,'IC %.6f n %d'%(q.mean(),len(q)))
print('coverage %.4f turnover %.4f'%(fac.notna().sum(axis=1).mean()/len(watch),fac.rank(axis=1,pct=True).diff().abs().mean().mean()))
