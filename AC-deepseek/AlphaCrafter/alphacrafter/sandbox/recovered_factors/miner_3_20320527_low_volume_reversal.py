import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}; V={}
for a in assets:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
 D[a]=d['close']; V[a]=d['volume']
P=pd.DataFrame(D).sort_index(); Vol=pd.DataFrame(V).reindex(P.index)
R=P.pct_change(); r3=P.pct_change(3)
# Novel idea: short-term reversal is stronger when the move occurred on unusually light volume.
# volume surprise is current 5d mean volume relative to trailing 60d median; use lagged completed data.
vs=Vol.rolling(5,min_periods=3).mean()/Vol.rolling(60,min_periods=30).median()-1
F=(-r3)/(1+vs.clip(lower=-.8))
# lag signal one day to enforce decision visibility
F=F.shift(1)
fr={h:P.shift(-h)/P-1 for h in [1,5,10,20]}
print('data',P.index.min(),P.index.max(),'assets',len(assets),'dates',len(P),'coverage',F.notna().mean().mean())
def calc(h):
 vals=[]; ns=[]
 for d in P.index:
  x=F.loc[d]; y=fr[h].loc[d]; z=x.notna()&y.notna()&np.isfinite(x)&np.isfinite(y)
  if z.sum()>=8 and x[z].nunique()>1 and y[z].nunique()>1:
   vals.append(spearmanr(x[z],y[z]).statistic); ns.append(z.sum())
 s=pd.Series(vals); print('h',h,'dates',len(s),'meanN',round(np.mean(ns),2),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4),'std',round(s.std(),5))
 return s
for h in [1,5,10,20]: calc(h)
# regime and recent stability
for lo,hi in [('2020','2023-12-31'),('2024','2027-12-31'),('2028','2030-12-31')]:
 idx=P.loc[lo:hi].index; vals=[]
 for d in idx:
  x=F.loc[d]; y=fr[1].loc[d]; z=x.notna()&y.notna()
  if z.sum()>=8 and x[z].nunique()>1 and y[z].nunique()>1: vals.append(spearmanr(x[z],y[z]).statistic)
 s=pd.Series(vals); print('regime',lo,hi,'dates',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std() if len(s)>1 else np.nan)
# proxy turnover
rank=F.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rank)):
 z=rank.iloc[i-1].notna()&rank.iloc[i].notna()
 if z.sum()>=8: ts.append((rank.iloc[i-1][z]-rank.iloc[i][z]).abs().mean())
print('turnover10_proxy',np.mean(ts[::10]))
