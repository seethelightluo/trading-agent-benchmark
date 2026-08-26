import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-06-27')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}; P=pd.concat(D,axis=1).loc[:cut]; L=np.log(P); R=L.diff(); ret10=L-L.shift(10); dates=P.index
out={h:[] for h in [1,5,10,20]}; cov=[]; turn=[]; prev=None; rows=[]
for i,t in enumerate(dates):
 if i<65: continue
 x=ret10.iloc[i]; valid=x.notna();
 if valid.sum()<8: continue
 vol=R.iloc[max(0,i-29):i+1].std(); med=x[valid].median(); f=(x-med)/vol
 valid=f.notna();
 if valid.sum()<8: continue
 ranks=f.rank(pct=True); cov.append(valid.sum()/15); turn.append(0 if prev is None else (ranks-prev.reindex(ranks.index).fillna(.5)).abs().mean()); prev=ranks
 for s in U:
  if valid.get(s,False): rows.append((t.date(),s,f[s]))
 for h in out:
  fw=L.shift(-h).iloc[i]-L.iloc[i]; z=pd.concat([f,fw],axis=1).dropna()
  if len(z)>=8: out[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('cutoff',cut.date(),'universe',len(U),'dates',len(dates),'valid_dates',len(cov),'coverage',np.mean(cov),'turnover',np.mean(turn))
for h,a in out.items():
 z=pd.Series(a);print('H',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_3_20320628_relative_strength_signal.csv',index=False)
