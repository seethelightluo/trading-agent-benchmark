import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end='2026-07-15'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:end] for s in U}; dates=D['SPX'].index; P=pd.DataFrame({s:D[s].close.reindex(dates) for s in U},index=dates); R=P.pct_change()
trend=(R.gt(0).rolling(12,min_periods=9).mean()-R.lt(0).rolling(12,min_periods=9).mean()).shift(1)
vm=(P.pct_change(15).div(R.rolling(30,min_periods=20).std(),axis=0)).shift(1)
def csrank(x): return x.rank(axis=1,pct=True)-.5
for a in [.25,.5,.75,1.0]:
 F=csrank(vm)+a*csrank(trend); q=[];ns=[];ds=[]
 Y=P.shift(-1).div(P).sub(1)
 for dt in dates:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.f,z.y).statistic);ns.append(len(z));ds.append(dt)
 q=np.asarray(q);print('blend',a,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4),'coverage',round(F.notna().sum().sum()/F.size,4))
 if a==.5:F.to_csv('scripts/miner_2_20260730_trend_vol_blend_signal.csv')
