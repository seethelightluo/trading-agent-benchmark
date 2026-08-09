import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date').close for a in A}
R={a:P[a].pct_change() for a in A}
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date')['close']
# VIX-conditioned reversal: short-term reversal is stronger when volatility is elevated, but
# use only lagged VIX percentile and sign; factor rewards reversal in high-vol regime.
vixrank=vix.rolling(252,min_periods=60).rank(pct=True).shift(1)
F={a:-R[a].rolling(5).sum()* (0.5+vixrank.reindex(P[a].index).fillna(0.5)) for a in A}
rows=[]; sig=[]
for d in sorted(set().union(*[set(x.index) for x in P.values()])):
 vals={a:F[a].get(d,np.nan) for a in A}; good=[v for v in vals.values() if np.isfinite(v)]
 if len(good)<8: continue
 med=np.nanmedian(good)
 for a in A: sig.append((d,a,vals[a]-med if np.isfinite(vals[a]) else np.nan))
 for h in [1,5,10]:
  f=[];y=[]
  for a in A:
   if d not in P[a].index: continue
   i=P[a].index.get_loc(d); z=vals[a]-med
   if i+h<len(P[a]) and np.isfinite(z): f.append(z); y.append(P[a].iloc[i+h]/P[a].iloc[i]-1)
  if len(f)>=8: rows.append((d,h,spearmanr(f,y).statistic,len(f)))
df=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 x=df[df.h==h]; print('H',h,'dates',len(x),'avg_n',round(x.n.mean(),2),'coverage',round(x.n.mean()/15,4),'IC',round(x.ic.mean(),6),'ICIR',round(x.ic.mean()/x.ic.std(),6),'hit',round((x.ic>0).mean(),4))
 for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026'),('2026-07','2027')]:
  z=x.set_index('date').loc[lo:hi].ic; print(lo,len(z),round(z.mean(),6),round(z.mean()/z.std(),6) if len(z)>1 else np.nan)
w=pd.DataFrame(sig,columns=['date','asset','signal']).pivot(index='date',columns='asset',values='signal'); print('turnover',round(w.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
pd.DataFrame(sig,columns=['date','asset','signal']).to_csv('../persistent/factor_signals_miner_3_20270225_vix_cond_reversal.csv',index=False)
