import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; C=pd.Timestamp('2026-07-15')
def ld(s): return pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date')['close'].sort_index()
px=pd.concat([ld(s).rename(s) for s in U],axis=1,join='inner').sort_index(); r=px.pct_change(); m=r['SPX']
# Market-residual momentum: 20d asset return less 60d rolling beta times 20d SPX return.
beta=r.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0)
f=px.pct_change(20).sub(beta.mul(px['SPX'].pct_change(20),axis=0),axis=0)
def ev(h):
 fw=px.shift(-h)/px-1; out=[]
 for d in f.index[f.index<=C]:
  z=pd.concat([f.loc[d],fw.loc[d]],axis=1).dropna()
  if len(z)>=8: out.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
a=ev(1); print('range',a.index.min(),a.index.max(),'dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15)); print('daily',a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean())
for h in [5,10]:
 q=ev(h); print('horizon',h,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=a[(a.index.year>=lo)&(a.index.year<=hi)]; print('regime',lo,hi,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'dates',len(q))
rank=f.loc[:C].rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean()); f.loc[:C].to_csv('scripts/miner_2_20260827_residual_momentum_signal.csv')
