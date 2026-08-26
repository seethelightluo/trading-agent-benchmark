import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2029-11-17'); base='../persistent/stock_data'
P=pd.concat([pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
r=np.log(P).diff()
# Trend consistency: cumulative 20d log return scaled by fraction of positive daily moves; lagged.
ret20=r.rolling(20,min_periods=15).sum(); consistency=(r.gt(0).rolling(20,min_periods=15).mean()-0.5).abs()+0.5
sig=(ret20*consistency).shift(1)
print('rows',len(P),'assets',P.notna().sum().min())
for h in [5,10,20,40]:
 f=np.log(P.shift(-h)/P); out=[]; ns=[]; dates=[]
 for dt in P.index:
  z=pd.concat([sig.loc[dt].rename('x'),f.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>1: out.append(spearmanr(z.x,z.y).statistic);ns.append(len(z));dates.append(dt)
 q=pd.Series(out,index=dates).dropna(); print('H',h,'dates',len(q),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
 for lab,lo,hi in [('early','2020-01-01','2026-12-31'),('mid','2027-01-01','2028-12-31'),('late','2029-01-01','2029-11-17')]:
  x=q.loc[lo:hi]; print(' ',lab,len(x),round(x.mean(),6),round(x.mean()/x.std(ddof=1),6) if len(x)>1 else np.nan)
rank=sig.rank(axis=1,pct=True); t=[]
for a,b in zip(rank.index[:-1],rank.index[1:]):
 z=pd.concat([rank.loc[a],rank.loc[b]],axis=1).dropna()
 if len(z):t.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print('turnover',round(float(np.mean(t)),6),'valid_dates',sig.dropna(how='all').shape[0])
sig.to_csv('scripts/miner_1_20291119_consistency20_signal.csv')
