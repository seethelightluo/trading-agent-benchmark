import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in assets}
p=pd.DataFrame(px).sort_index(); r=p.pct_change(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill();
# One interpretable candidate: macro-confirmed momentum, cross-sectional demeaned risk-adjusted 20d trend,
# attenuated (not inverted) when VIX is rising, lagged one day.
raw=p.pct_change(20)/r.rolling(20).std(); cs=raw.sub(raw.mean(axis=1),axis=0)
reg=(1/(1+vix.pct_change(10).clip(lower=0)*4)).replace([np.inf,-np.inf],np.nan)
f=cs.mul(reg,axis=0).shift(1)
rows=[]
for d in f.index:
 x=f.loc[d]; y=p.pct_change(5).shift(-5).loc[d]
 z=pd.concat([x,y],axis=1).dropna();
 if len(z)>=8:
  rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z),len(z)/15))
out=pd.DataFrame(rows,columns=['date','ic','n','coverage']).set_index('date')
for recent in [None,500,250]:
 q=out if recent is None else out.tail(recent); print('window',recent,'dates',len(q),'avgN',q.n.mean(),'coverage',q.coverage.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit', (q.ic>0).mean())
for a,b in [('2026','2029'),('2030','2033'),('2034','2035')]:
 q=out.loc[a:b];print(a,b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1))
# turnover rank ordering proxy
rank=f.rank(axis=1,pct=True); print('turnover',rank.diff().abs().mean().mean(),'valid dates',len(out))
out.to_csv('scripts/miner_1_20351001_macro_confirmed_momentum_signal.csv')
print('artifact written')
