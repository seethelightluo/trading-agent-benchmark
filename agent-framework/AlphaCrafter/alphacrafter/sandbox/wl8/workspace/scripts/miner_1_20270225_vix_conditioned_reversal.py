import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
p=pd.concat(P,axis=1).sort_index(); r=p.pct_change(); v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill()
# Reversal is stronger in calm markets; in stress, dampen reversal to avoid fighting broad shocks.
stress=(v>v.rolling(60,min_periods=30).median()).astype(float)
f=-r.rolling(5,min_periods=5).sum().mul(1-0.6*stress,axis=0)
ics=[]; ns=[]; dates=[]; turns=[]; prev=None
for i in range(65,len(p)-1):
 z=pd.concat([f.iloc[i],r.iloc[i+1]],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(p.index[i])
  if prev is not None: turns.append(np.mean(np.sign(f.iloc[i].reindex(U))!=np.sign(prev.reindex(U))))
  prev=f.iloc[i]
a=np.array(ics); print('dates',len(a),'avg_n',np.mean(ns),'coverage',sum(ns)/(len(ns)*15),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',(a>0).mean(),'turn',np.mean(turns))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
 b=a[[x.year>=lo and x.year<=hi for x in dates]]; print('regime',lo,hi,'n',len(b),'IC',b.mean(),'ICIR',b.mean()/b.std(ddof=1) if len(b)>1 else np.nan)
