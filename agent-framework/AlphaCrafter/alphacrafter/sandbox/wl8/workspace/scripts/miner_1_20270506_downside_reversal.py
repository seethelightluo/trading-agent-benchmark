import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(P,axis=1).sort_index(); r=p.pct_change()
# Downside-risk-adjusted short reversal: penalize recent losses by downside semideviation,
# using only completed trailing observations.
down=r.where(r<0,0).rolling(20,min_periods=10).std()
f=-r.rolling(3,min_periods=3).sum()/(down+1e-6)
ics=[];ns=[];dates=[];turn=[];prev=None
for i in range(65,len(p)-1):
 z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(p.index[i])
  if prev is not None: turn.append(np.mean(np.sign(f.iloc[i].reindex(U))!=np.sign(prev.reindex(U))))
  prev=f.iloc[i]
a=np.array(ics);print('dates',len(a),'rows',sum(ns),'avg_n',np.mean(ns),'coverage',sum(ns)/(15*len(ns)),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'turn',np.mean(turn))
for lo,hi in [(2020,2022),(2023,2025),(2026,2026),(2027,2027)]:
 b=a[[lo<=x.year<=hi for x in dates]];print('regime',lo,hi,'n',len(b),'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1))
