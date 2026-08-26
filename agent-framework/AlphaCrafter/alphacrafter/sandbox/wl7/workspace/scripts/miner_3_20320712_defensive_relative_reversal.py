import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; DEF=['XAU','US10Y','CN10Y']; cut=pd.Timestamp('2032-07-10')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}; P=pd.concat(D,axis=1).loc[:cut]; L=np.log(P); R=L.diff(); r5=L-L.shift(5); dates=P.index
outs={h:[] for h in [1,5,10,20]}; cov=[]; turn=[]; prev=None; rows=[]
for i,t in enumerate(dates):
 if i<35: continue
 x=r5.iloc[i]; dv=x[DEF].mean(); vol=R.iloc[i-19:i+1].std(); f=-(x-dv)/vol
 valid=f.notna()
 if valid.sum()<8: continue
 ranks=f.rank(pct=True); cov.append(valid.sum()/15); turn.append(0 if prev is None else (ranks-prev.reindex(ranks.index).fillna(.5)).abs().mean()); prev=ranks
 for s in U:
  if valid[s]: rows.append((t.date(),s,float(f[s])))
 for h in outs:
  fw=L.shift(-h).iloc[i]-L.iloc[i]; z=pd.concat([f,fw],axis=1).dropna()
  if len(z)>=8: outs[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('cutoff',cut.date(),'universe',len(U),'dates',len(dates),'valid_dates',len(cov),'avgN',np.mean(cov)*15,'coverage',np.mean(cov),'turnover',np.mean(turn))
for h,a in outs.items():
 z=pd.Series(a); print('H',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_3_20320712_defensive_relative_reversal_signal.csv',index=False)
