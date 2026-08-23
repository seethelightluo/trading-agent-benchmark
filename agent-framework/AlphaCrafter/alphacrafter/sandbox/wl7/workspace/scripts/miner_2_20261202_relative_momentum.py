import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2026-12-01')
px={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'
 if not os.path.exists(p): p=f'../persistent/index_data/{s}.csv'
 d=pd.read_csv(p); d['date']=pd.to_datetime(d['date']); d=d[d.date<=cut].set_index('date').sort_index()
 px[s]=d['close']
prices=pd.DataFrame(px).sort_index()
# relative medium momentum: 20d return less contemporaneous cross-sectional median, all inputs lagged one day
r20=prices.shift(1)/prices.shift(21)-1
fac=r20.sub(r20.median(axis=1),axis=0)
fwd=prices.shift(-1)/prices-1
ics=[]; turnovers=[]; ns=[]
for dt in fac.index:
 x=fac.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
# rank turnover
ranks=fac.rank(axis=1,pct=True); turnovers=ranks.diff().abs().mean(axis=1).dropna()
a=np.array(ics); print({'factor':'relative_20d_momentum','cutoff':str(cut.date()),'dates':len(a),'avg_instruments':round(np.mean(ns),3),'ic':round(a.mean(),6),'icir':round(a.mean()/a.std(ddof=1),6),'hit':round((a>0).mean(),4),'coverage':round(np.mean(ns)/15,5),'turnover':round(turnovers.mean(),6)})
for h in [5,10,20]:
 ff=prices.shift(-h)/prices-1; q=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],ff.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q); print(h,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
# regimes
for a0,b0 in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=[]
 for dt in fac.index:
  if a0<=str(dt.year)<=b0:
   z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
   if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q); print(a0,b0,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
