import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); lr=np.log(P).diff()
# Range-location persistence: assets near the upper end of their 40d range with
# positive 20d trend receive higher scores; volatility scales conviction.
hi=P.rolling(40).max(); lo=P.rolling(40).min()
loc=(P-lo)/(hi-lo+1e-12)*2-1
mom=P/P.shift(20)-1
vol=lr.rolling(20).std()*np.sqrt(252)
f=(loc*mom/(vol+1e-12)).shift(1)

def calc(h):
 rows=[]
 for i in range(len(P)-h):
  fut=P.iloc[i+h]/P.iloc[i]-1; x=f.iloc[i]; ok=x.notna()&fut.notna()
  if ok.sum()>=8: rows.append((P.index[i],spearmanr(x[ok],fut[ok]).statistic,ok.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 return z
z=calc(10)
print('universe',len(syms),'available',len(P.columns),'dates',len(z),'avg_names',z.n.mean(),'IC',z.ic.mean(),'ICIR',z.ic.mean()/z.ic.std(ddof=1),'hit',(z.ic>0).mean())
for h in [1,5,10,20,40]:
 q=calc(h); print('decay',h,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'dates',len(q))
for a,b in [('2020','2024-12-31'),('2025','2027-12-31'),('2028','2030-04-01')]:
 q=z.loc[a:b]; print('regime',a,b,'dates',len(q),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1) if len(q)>1 else np.nan)
rank=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turnover',rank.diff().abs().mean().mean())
z.to_csv('scripts/miner_3_20300408_range_persistence_ic.csv'); f.to_csv('scripts/miner_3_20300408_range_persistence_signal.csv')
