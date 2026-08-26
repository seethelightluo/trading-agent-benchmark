import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2032-07-11')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}
P=pd.concat(D,axis=1).loc[:cut]; L=np.log(P); R=L.diff(); dates=P.index
out={h:[] for h in [1,5,10,20]}; cov=[]; turn=[]; prev=None; rows=[]; thirds={h:[[] for _ in range(3)] for h in out}
for i,t in enumerate(dates):
 if i<65: continue
 # lag-safe volatility-normalized 20d relative strength; cross-sectional median removal
 x=L.iloc[i]-L.shift(20).iloc[i]; vol=R.iloc[max(0,i-39):i].std()
 f=x/vol; valid=f.notna()
 if valid.sum()<8: continue
 f=f-f[valid].median(); valid=f.notna()
 if valid.sum()<8: continue
 ranks=f.rank(pct=True); cov.append(valid.sum()/15); turn.append(0 if prev is None else (ranks-prev.reindex(ranks.index).fillna(.5)).abs().mean()); prev=ranks
 for s in U:
  if valid.get(s,False): rows.append((t.date(),s,float(f[s])))
 for h in out:
  fw=L.shift(-h).iloc[i]-L.iloc[i]; z=pd.concat([f,fw],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; out[h].append(ic)
   thirds[h][min(2,int(len(out[h])*3/len(dates)) )].append(ic)
print('cutoff',cut.date(),'universe',len(U),'calendar_dates',len(dates),'valid_dates',len(cov),'avgN',np.mean(cov)*15,'coverage',np.mean(cov),'turnover',np.mean(turn))
for h,a in out.items():
 z=pd.Series(a); print('H',h,'dates',len(z),'IC',z.mean(),'ICIR',z.mean()/z.std(ddof=1),'hit',(z>0).mean())
print('recent thirds H10')
# chronological thirds by observation
for k,a in enumerate(np.array_split(np.array(out[10]),3)):
 q=pd.Series(a); print(k+1,len(q),q.mean() if len(q) else np.nan,(q.mean()/q.std(ddof=1)) if len(q)>1 else np.nan)
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20320712_volnorm_relative_strength_signal.csv',index=False)
