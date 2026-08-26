import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-07-11')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}; P=pd.concat(D,axis=1).loc[:cut]; L=np.log(P); R=L.diff(); dates=P.index
out={1:[],5:[],10:[],20:[]}; cov=[]; turns=[]; prev=None; rows=[]
for i,t in enumerate(dates):
 if i<65: continue
 r5=L.iloc[i]-L.shift(5).iloc[i]; v20=R.iloc[max(0,i-20):i].std(); v60=R.iloc[max(0,i-60):i].std()
 # reversal is preferred in high-volatility expansion, momentum in compression
 ratio=(v20/v60).clip(.5,2.0); f=(-r5/v20)*ratio + (L.iloc[i]-L.shift(20).iloc[i])/v20*(2-ratio)
 valid=f.notna()
 if valid.sum()<8: continue
 f=f-f[valid].median(); ranks=f.rank(pct=True); cov.append(valid.sum()/15)
 turns.append(0 if prev is None else (ranks-prev.reindex(ranks.index).fillna(.5)).abs().mean()); prev=ranks
 for s in U:
  if valid.get(s,False): rows.append((t.date(),s,float(f[s])))
 for h in out:
  fw=L.shift(-h).iloc[i]-L.iloc[i]; z=pd.concat([f,fw],axis=1).dropna()
  if len(z)>=8: out[h].append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('cutoff',cut.date(),'universe',len(U),'calendar_dates',len(dates),'valid_dates',len(cov),'avgN',np.mean(cov)*15,'coverage',np.mean(cov),'turnover',np.mean(turns))
for h,a in out.items():
 z=pd.Series(a); print('H',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
for k,a in enumerate(np.array_split(np.array(out[10]),3)):
 z=pd.Series(a); print('H10third',k+1,len(z),z.mean(),z.mean()/z.std(ddof=1))
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20320712_regime_switch_signal.csv',index=False)
