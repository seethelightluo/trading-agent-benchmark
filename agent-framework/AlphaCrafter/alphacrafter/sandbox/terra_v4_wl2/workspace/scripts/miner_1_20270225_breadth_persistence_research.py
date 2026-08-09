import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P='../persistent/stock_data/'; CUT=pd.Timestamp('2027-02-25')
C=pd.concat({s:pd.read_csv(P+s+'.csv',parse_dates=['date']).set_index('date')['close'] for s in U},axis=1).sort_index(); C=C.loc[C.index<=CUT]
r=C.pct_change(); breadth=(r<0).sum(axis=1)/r.notna().sum(axis=1); rr=C.pct_change(3).shift(1); base=-rr.sub(rr.median(axis=1),axis=0); fwd=C.shift(-5)/C-1
hist=breadth.shift(1).rolling(252,min_periods=60); variants={}
for n in [1,2,3]:
 for qname,q in [('med',hist.median()),('q75',hist.quantile(.75))]:
  flag=breadth.shift(1)>=np.maximum(.60,q); variants[f'p{n}_{qname}']=flag.rolling(n,min_periods=n).sum().eq(n)
for name,active in variants.items():
 vals=[]; dates=[]; ns=[]
 for dt in C.index:
  if not bool(active.get(dt,False)): continue
  z=pd.concat([base.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); dates.append(dt); ns.append(len(z))
 a=np.asarray(vals); ic=np.mean(a) if len(a) else np.nan; sd=np.std(a,ddof=1) if len(a)>1 else np.nan
 print(name,'dates',len(a),'avgN',np.mean(ns) if ns else 0,'IC %.6f ICIR %.6f hit %.4f'%(ic,ic/sd if sd else np.nan,np.mean(a>0) if len(a) else np.nan))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2027)]:
  aa=np.array([v for v,d in zip(vals,dates) if lo<=d.year<=hi]); print(' ',lo,hi,'n',len(aa),'ic',np.mean(aa) if len(aa) else np.nan)
print('range',C.index.min(),C.index.max(),'dates',len(C),'instruments',C.shape[1])
