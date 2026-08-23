import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[a]=d
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].sort_index().reindex(p.index).ffill()
# causal rolling percentile (expanding minimum history)
vrank=vix.rolling(252,min_periods=60).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1])
# adaptive: momentum in calm, contrarian in stressed; cross-sectional scalar regime switch
mom=p.pct_change(20)/r.rolling(40).std()
sig=mom.where(vrank<0.75,-mom)
# median demean preserves cross-sectional signal
sig=sig.sub(sig.median(axis=1),axis=0)
rows=[]
for d in p.index:
 if d>pd.Timestamp('2030-06-26'): break
 f=sig.loc[d]; fr=p.shift(-10).loc[d]/p.loc[d]-1
 z=pd.concat([f,fr],axis=1).dropna()
 if len(z)>=8:
  rows.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
ic=np.array([x[1] for x in rows]); print('dates',len(ic),'mean_n',p.loc[[x[0] for x in rows]].notna().sum(axis=1).mean(),'IC',ic.mean(),'ICIR',ic.mean()/ic.std(ddof=1),'hit',(ic>0).mean())
for lo,hi in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-06-26')]:
 q=[v for d,v in rows if d>=pd.Timestamp(lo) and d<=pd.Timestamp(hi)]; q=np.array(q); print(lo,hi,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# horizons
for h in [5,10,20]:
 vals=[]
 for d in p.index:
  if d>pd.Timestamp('2030-06-26'): break
  z=pd.concat([sig.loc[d],p.shift(-h).loc[d]/p.loc[d]-1],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(vals); print('h',h,'n',len(a),'ic',a.mean(),'icir',a.mean()/a.std(ddof=1))
# artifact
sig.loc[:'2030-06-26'].to_csv('scripts/miner_2_20300627_adaptive_momentum_signal.csv')
print('coverage',sig.loc[:'2030-06-26'].notna().stack().mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
