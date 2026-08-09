import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').sort_values('date').set_index('date').close.astype(float) for a in A}
R={a:P[a].pct_change() for a in A}
# low-volatility anomaly: inverse 30d realized vol, lagged one completed day
S={a:-(R[a].rolling(30,min_periods=20).std()) for a in A}
rows=[]
for d in sorted(set().union(*[set(x.index) for x in P.values()])):
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if d not in P[a].index or not np.isfinite(S[a].get(d,np.nan)): continue
   ix=P[a].index.get_loc(d)
   if ix+h>=len(P[a]): continue
   yy=P[a].iloc[ix+h]/P[a].iloc[ix]-1
   if np.isfinite(yy): f.append(S[a].loc[d]);y.append(yy)
  if len(f)>=8: rows.append((d,h,spearmanr(f,y).statistic,len(f)))
out=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=out[out.h==h]; print('H',h,'dates',len(x),'avg_names',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4),'turnover',round(pd.DataFrame({a:S[a] for a in A}).rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2027','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),6) if len(z) else np.nan,round(z.mean()/z.std(),6) if len(z)>1 else np.nan)
sig=pd.DataFrame([(d,a,S[a].get(d,np.nan)) for d in sorted(set().union(*[set(x.index) for x in P.values()])) for a in A],columns=['date','asset','signal'])
sig.to_csv('../persistent/factor_signals_miner_2_20270225_lowvol30.csv',index=False)
print('artifact',sig.signal.notna().sum(),'dates',sig.date.nunique(),'assets',sig.groupby('date').asset.count().mean())
