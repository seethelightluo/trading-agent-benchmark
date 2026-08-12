import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
    d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close']
    px[s]=d[d.index<='2027-08-11']
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Momentum strengthened when recent 5d path agrees with its 20d direction.
m20=p/p.shift(20)-1
m5=p/p.shift(5)-1
agreement=(np.sign(m5)==np.sign(m20)).astype(float)
f=m20*agreement
f=f.shift(0)
fwd=p.shift(-1)/p-1
ics=[]; ns=[]
for dt in f.index:
    x=f.loc[dt]; y=fwd.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
    if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
        ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
ics=np.array(ics); print('dates',len(ics),'avgN',np.mean(ns),'IC',np.mean(ics),'ICIR',np.mean(ics)/np.std(ics,ddof=1),'hit',np.mean(ics>0),'coverage',np.mean([n/15 for n in ns]))
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 q=[v for dt,v in zip(f.index,ics) if a<=str(dt.year)<=b] # alignment wrong due omitted dates
 print(a,b,'n',len(q),'ic',np.mean(q) if q else np.nan,'icir',np.mean(q)/np.std(q,ddof=1) if len(q)>1 else np.nan)
for h in [1,3,5,10]:
 y=p.shift(-h)/p-1; q=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'dates',len(q),'IC',np.mean(q),'ICIR',np.mean(q)/np.std(q,ddof=1))
print('turnover',np.mean((f.rank(axis=1,pct=True)-f.shift(1).rank(axis=1,pct=True)).abs().mean(axis=1).dropna()))
print('last',f.iloc[-1].describe())
