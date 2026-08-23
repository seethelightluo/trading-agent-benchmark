import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-01'); O={};C={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'; d=pd.read_csv(p);d.date=pd.to_datetime(d.date);d=d[d.date<=cut].set_index('date').sort_index(); O[s]=d.open;C[s]=d.close
op=pd.DataFrame(O).sort_index(); cl=pd.DataFrame(C).sort_index()
# lagged short-horizon reversal weighted by prior overnight gap: fade close-to-close move, but reward close near extreme
r3=cl.shift(1)/cl.shift(4)-1
rng=(cl.shift(1)*0+1) # keep alignment
# robust volatility normalization
rv=cl.pct_change().rolling(20).std().shift(1)
fac=-r3/(rv+1e-8)
fwd=cl.shift(-1)/cl-1
all=[]; ns=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8:all.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
a=np.array(all);turn=fac.rank(pct=True).diff().abs().mean(axis=1).dropna()
print({'factor':'vol_scaled_3d_reversal','cutoff':str(cut.date()),'dates':len(a),'avg_instruments':round(np.mean(ns),3),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1),6),'hit':round((a>0).mean(),4),'coverage':round(np.mean(ns)/15,5),'turnover':round(turn.mean(),6)})
for h in [5,10,20]:
 q=[]; ff=cl.shift(-h)/cl-1
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print(h,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
for ya,yb in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=[]
 for dt in fac.index:
  if ya<=str(dt.year)<=yb:
   z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print(ya,yb,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
