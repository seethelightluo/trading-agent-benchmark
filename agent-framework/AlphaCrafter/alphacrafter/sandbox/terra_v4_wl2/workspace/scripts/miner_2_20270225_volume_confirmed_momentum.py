import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={};V={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').drop_duplicates('date').set_index('date');P[a]=d.close.astype(float);V[a]=d.volume.astype(float)
R={a:P[a].pct_change() for a in A}
# interpretable volume-confirmed medium momentum: trailing 5d return times log volume surprise
S={a:(P[a]/P[a].shift(5)-1)*np.log((V[a]+1)/(V[a].rolling(20,min_periods=10).median()+1)) for a in A}
rows=[]
for d in sorted(set().union(*[set(x.index) for x in P.values()])):
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if d not in P[a].index or not np.isfinite(S[a].get(d,np.nan)):continue
   i=P[a].index.get_loc(d)
   if i+h>=len(P[a]):continue
   yy=P[a].iloc[i+h]/P[a].iloc[i]-1
   if np.isfinite(yy):f.append(S[a].loc[d]);y.append(yy)
  if len(f)>=8:rows.append((d,h,spearmanr(f,y).statistic,len(f)))
out=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=out[out.h==h];print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic;print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if len(z)>1 else np.nan)
sig=pd.DataFrame([(d,a,S[a].get(d,np.nan)) for d in sorted(set().union(*[set(x.index) for x in P.values()])) for a in A],columns=['date','asset','signal']);sig.to_csv('../persistent/factor_signals_miner_2_20270225_volume_confirmed_momentum.csv',index=False)
w=sig.pivot(index='date',columns='asset',values='signal');print('turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6),'signal_dates',w.notna().any(axis=1).sum())
