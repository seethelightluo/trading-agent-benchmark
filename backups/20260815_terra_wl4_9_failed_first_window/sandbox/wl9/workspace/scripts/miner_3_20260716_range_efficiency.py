import pandas as pd, numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base=Path('../persistent/stock_data'); end=pd.Timestamp('2026-07-15')
px={}
for s in U:
 d=pd.read_csv(base/f'{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d[d.index<=end]
P=pd.DataFrame(px).sort_index().ffill(); R=P.pct_change()
# range efficiency: signed net return divided by path variation, positive means persistent upward path
f=R.rolling(20).sum()/(R.abs().rolling(20).sum()+1e-8)
# forward 1/5/10 day close returns, observations with >=8 names
for h in [1,5,10]:
 y=P.shift(-h)/P-1; vals=[]; dates=[]; ns=[]
 for dt in P.index:
  a=f.loc[dt]; b=y.loc[dt]; z=pd.concat([a,b],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
 x=np.array(vals); print('h',h,'obs',len(x),'meanN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/(x.std(ddof=1)+1e-12),'hit',(x>0).mean())
# rank turnover and coverage
ranks=f.rank(axis=1,pct=True); turn=(ranks-ranks.shift(1)).abs().mean(axis=1).dropna().mean()
print('dates',len(P),'instruments',len(U),'coverage',f.notna().sum(axis=1).ge(8).mean(),'turnover',turn)
# regime halves and recent
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 sub=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],(P.shift(-1)/P-1).loc[dt]],axis=1).dropna()
  if len(z)>=8: sub.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print(lo,hi,'n',len(sub),'IC',np.mean(sub) if sub else np.nan)
