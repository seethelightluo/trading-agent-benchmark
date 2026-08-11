import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
U=get_account_dict().get('watch_list') or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is None or len(d)<150:d=get_index_daily_data(s,days=3000)
 if d is not None and len(d)>150:
  d['date']=pd.to_datetime(d['date']);P[s]=d.set_index('date')['close'].sort_index()
px=pd.concat(P,axis=1).sort_index().ffill();r=px.pct_change(); mr=r.mean(axis=1)
# Smooth 3-day reversal, activated only in objectively stressed cross-asset regimes.
# Stress percentile is computed from prior observations and whole factor is lagged.
rv=-r.rolling(3,min_periods=3).sum(); v=mr.rolling(20,min_periods=15).std(); base=mr.rolling(120,min_periods=60).std(); ratio=v/base
thr=ratio.rolling(252,min_periods=100).quantile(.65)
f=rv.where(ratio.gt(thr),0.0).shift(1)
for h in [1,3,5,10]:
 vals=[];ns=[];fr=px.pct_change(h).shift(-h)
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q):vals.append(q);ns.append(len(a))
 x=pd.Series(vals);print('H',h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).mean()/len(U),4),'turnover',round((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)>0.05).mean(),4))
for lab,aa in [('2020-22',x.loc['2020':'2022']),('2023-25',x.loc['2023':'2025']),('2026-27',x.loc['2026':'2027']),('2028YTD',x.loc['2028':])]:print(lab,len(aa),round(aa.mean(),6),round(aa.mean()/aa.std(ddof=1),6) if len(aa)>1 else np.nan)
