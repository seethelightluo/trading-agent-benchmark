import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; p={}
for a in A:
 d=pd.read_csv(f'{base}/{a}.csv',parse_dates=['date']).set_index('date'); p[a]=d.close.astype(float)
px=pd.DataFrame(p).sort_index().loc[:'2027-03-16']; ret=px.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.astype(float).reindex(px.index).ffill()
vz=(v-v.rolling(60,min_periods=30).mean())/v.rolling(60,min_periods=30).std()
f=(-px.pct_change(5).shift(1)/ret.rolling(20,min_periods=15).std().shift(1)).mul((1+.35*np.tanh(vz.shift(1).fillna(0))).clip(.65,1.35),axis=0)
for h in [1,5,10]:
 y=px.shift(-h)/px-1; ic=[]; ns=[]; dates=[]
 for dt in px.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 s=pd.Series(ic); print(h,len(s),np.mean(ns),s.mean(),s.mean()/s.std(ddof=1)*np.sqrt(252),(s>0).mean())
 if h==1: out=pd.DataFrame({'date':dates,'signal_ic':ic})
print('assets',len(p),'rows',len(px),'coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
out.to_csv('scripts/miner_1_20270317_vix_revalidation_signal.csv',index=False)
for label,sub in out.assign(year=pd.to_datetime(out.date).dt.year).groupby(pd.cut(out.assign(year=pd.to_datetime(out.date).dt.year).year,[2019,2022,2024,2030],labels=['2020-22','2023-24','2025+'])): print(label,len(sub),sub.signal_ic.mean())
