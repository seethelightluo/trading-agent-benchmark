import pandas as pd,numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; DEF=['XAU','US10Y','CN10Y']; cut=pd.Timestamp('2032-07-10')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}; P=pd.concat(D,axis=1).loc[:cut]; L=np.log(P); dates=P.index
rows=[]; prev=None; turns=[]; cov=[]
for i,t in enumerate(dates):
 if i<7: continue
 x=(L-L.shift(5)).iloc[i]; f=-(x-x[DEF].median()); valid=f.notna()
 if valid.sum()<8: continue
 ranks=f.rank(pct=True); cov.append(valid.sum()/15)
 if prev is not None: turns.append((ranks-prev.reindex(ranks.index).fillna(.5)).abs().mean())
 prev=ranks
 for s in U:
  if valid[s]: rows.append((t.date(),s,float(f[s])))
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_3_20320712_raw_defensive_reversal_5d_signal.csv',index=False)
print('artifact rows',len(rows),'dates',len(cov),'coverage',np.mean(cov),'turnover',np.mean(turns))
