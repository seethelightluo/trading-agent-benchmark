import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-12-01'); P={}
for s in U:
 x=pd.read_csv('../persistent/stock_data/'+s+'.csv');x.date=pd.to_datetime(x.date);P[s]=x[x.date<=END].set_index('date').close.sort_index()
px=pd.DataFrame(P).sort_index(); mm=[]
for s in ['DXY','USDCNY','USDJPY','EURUSD','VIX']:
 x=pd.read_csv('../persistent/index_data/'+s+'.csv');x.date=pd.to_datetime(x.date);mm.append(x[x.date<=END].set_index('date').close.sort_index().reindex(px.index).ffill().pct_change())
m=pd.concat(mm,axis=1); z=(m-m.rolling(60,min_periods=30).mean())/m.rolling(60,min_periods=30).std(); w=z.abs().mean(axis=1).shift(1); w=w.rolling(252,min_periods=100).rank(pct=True).shift(1).clip(0.1,1)
sig=-px.pct_change().shift(1).mul(w,axis=0); fwd=px.shift(-1)/px-1
vals=[];ns=[]
for d in px.index:
 g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
 if len(g)>=8 and g.s.nunique()>1:
  q=spearmanr(g.s,g.f).statistic
  if np.isfinite(q):vals.append(q);ns.append(len(g))
a=np.array(vals);print('dates',len(a),'avgN',np.mean(ns),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'coverage',sig.notna().sum().sum()/sig.size)
for y in [2020,2021,2022,2023,2024,2025,2026,2027]:
 b=np.array([v for v,d in zip(vals,[d for d in px.index if d in px.index])]) if False else None
 ds=[d for d in px.index if d in px.index]
 # recompute selected
 vv=[]
 for d in px.index[px.index.year==y]:
  g=pd.DataFrame({'s':sig.loc[d],'f':fwd.loc[d]}).dropna()
  if len(g)>=8 and g.s.nunique()>1: vv.append(spearmanr(g.s,g.f).statistic)
 print(y,len(vv),np.mean(vv) if vv else np.nan, (np.mean(vv)/np.std(vv,ddof=1)) if len(vv)>1 else np.nan)
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20271202_macro_stress_weighted_signal.csv',index=False)
