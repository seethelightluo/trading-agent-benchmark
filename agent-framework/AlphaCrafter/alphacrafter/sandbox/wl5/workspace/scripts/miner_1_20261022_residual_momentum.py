import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END='2026-10-21'
def load(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').close
p=pd.concat({s:load(s) for s in U},axis=1).sort_index(); R=p.pct_change(); n=len(p); fac=np.full((n,15),np.nan)
for j,s in enumerate(U):
 bms=[x for x in ['SPX','XAU'] if x!=s]
 for i in range(70,n):
  hist=R.iloc[:i][[s]+bms].dropna().tail(60)
  if len(hist)<40: continue
  X=np.c_[np.ones(len(hist)),hist[bms].values]; b=np.linalg.lstsq(X,hist[s].values,rcond=None)[0]
  q=R.iloc[:i][[s]+bms].dropna().tail(10)
  if len(q)>=8: fac[i,j]=np.sum(q[s].values-b[0]-q[bms].values@b[1:])
def calc(h):
 fw=p.pct_change(h).shift(-h); vals=[]; ns=[]; ds=[]
 for i in range(n):
  ok=np.isfinite(fac[i])&np.isfinite(fw.iloc[i].values)
  if ok.sum()>=8 and len(np.unique(fac[i,ok]))>1: vals.append(spearmanr(fac[i,ok],fw.iloc[i].values[ok]).statistic);ns.append(ok.sum());ds.append(p.index[i])
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  q=a[[lo<=d.year<=hi for d in ds]]; print('regime',lo,hi,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
for h in [1,5,10]: calc(h)
rank=pd.DataFrame(fac,index=p.index).rank(axis=1,pct=True); print('coverage',round(np.isfinite(fac).mean(),4),'turnover',round(rank.diff().abs().mean(axis=1).mean(),4),'period',p.index[0].date(),p.index[-1].date())
out=pd.DataFrame(fac,index=p.index,columns=U).stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_1_20261022_residual_momentum_signal.csv',index=False)
