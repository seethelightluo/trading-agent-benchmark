import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U}
pd_=pd.DataFrame(p).sort_index(); pd_=pd_[pd_.index<=pd.Timestamp('2027-08-11')]; ret=pd_.pct_change()
# Persistence/efficiency: directional 20d move divided by path length, gated by positive 5d direction.
move=pd_/pd_.shift(20)-1
path=ret.abs().rolling(20).sum()
f=(move/(path+1e-8)) * (np.sign(move)==np.sign(pd_/pd_.shift(5)-1)).astype(float)
print('range',pd_.index.min(),pd_.index.max())
for h in [1,3,5,10]:
 y=pd_.shift(-h)/pd_-1; q=[]; dates=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);dates.append(dt);ns.append(len(z))
 q=np.array(q); print('h',h,'dates',len(q),'avgN',np.mean(ns),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',np.mean(q>0))
 if h==1:
  for start,end in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
   a=np.array([v for d,v in zip(dates,q) if start<=str(d.year)<=end]); print(start,end,len(a),a.mean() if len(a) else np.nan,a.mean()/a.std(ddof=1) if len(a)>1 else np.nan)
print('coverage',f.notna().mean().mean(),'turnover',((f.rank(pct=True)-f.shift().rank(pct=True)).abs().mean(axis=1)).mean())
