import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,6000)
 if d is None: d=get_index_daily_data(s,6000)
 D[s]=d
prices=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index()
# Candidate: volatility-normalized short-horizon reversal, causal and cross-sectional
ret5=prices.pct_change(5); vol20=prices.pct_change().rolling(20).std();
raw=-ret5/vol20.replace(0,np.nan)
# lag one day; forward 10d return
sig=raw.shift(1); fwd=prices.shift(-10)/prices-1
rows=[]; daily=[]
for dt in sig.index:
 x=sig.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
  daily.append((dt,ic,len(z)))
  for s in z.index: rows.append((dt,s,x[s]))
di=pd.DataFrame(daily,columns=['date','ic','n']).set_index('date')
print('dates',len(di),'avgN',di.n.mean(),'coverage',len(rows)/(len(di)*15))
print('IC',di.ic.mean(),'ICIR',di.ic.mean()/di.ic.std(),'hit', (di.ic>0).mean())
# turnover rank changes
S=pd.DataFrame(rows,columns=['date','symbol','sig']).pivot(index='date',columns='symbol',values='sig').rank(axis=1,pct=True)
turn=S.diff().abs().mean(axis=1).dropna().mean(); print('turnover',turn)
for h in [5,10,20,40]:
 yy=prices.shift(-h)/prices-1; vals=[]
 for dt in sig.index:
  q=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(q)>=8: vals.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,np.nanmean(vals))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032'),('2033','2035')]:
 q=di.loc[a:b,'ic']; print('regime',a,b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
S.to_csv('scripts/miner_2_20350706_volnorm_reversal_signal.csv')
di.to_csv('scripts/miner_2_20350706_volnorm_reversal_ic.csv')
