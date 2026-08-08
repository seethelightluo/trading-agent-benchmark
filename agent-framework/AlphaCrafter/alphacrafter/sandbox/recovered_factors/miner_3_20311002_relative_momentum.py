import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'; d=pd.read_csv(f)
 date=[c for c in d.columns if c.lower() in ('date','datetime','time')][0]
 close=[c for c in d.columns if c.lower() in ('close','adj_close','adj close')][0]
 px[a]=d.assign(Date=pd.to_datetime(d[date]))[['Date',close]].rename(columns={close:a}).set_index('Date')[a]
p=pd.concat(px,axis=1).sort_index().loc[:'2031-10-02']
r=np.log(p/p.shift(1))
# relative momentum: return vs contemporaneous cross-sectional median over lookback
for look in [5,10,20,40,60]:
 f=r.rolling(look).sum().sub(r.rolling(look).sum().median(axis=1),axis=0)
 print('\nLOOK',look)
 for h in [1,5,10,20]:
  vals=[]; ns=[]
  for i in range(look+1,len(p)-h):
   x=f.iloc[i]; y=r.iloc[i+1:i+1+h].sum()
   z=pd.concat([x,y],axis=1).dropna()
   if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  a=np.array(vals); print(h,'dates',len(a),'N',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)))
 # turnover 10d rank proxy
 q=f.rank(axis=1,pct=True); print('coverage',f.notna().mean().mean(),'turn10',q.diff(10).abs().mean().mean())
 # regimes H10
 h=10; vals=[]; ds=[]
 for i in range(look+1,len(p)-h):
  z=pd.concat([f.iloc[i],r.iloc[i+1:i+1+h].sum()],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ds.append(p.index[i])
 for st,en in [('2020','2023'),('2024','2027'),('2028','2030'),('2031','2031')]:
  a=np.array([v for v,d in zip(vals,ds) if st<=str(d.year)<=en]);
  print('reg',st,en,len(a), 'IC %.5f ICIR %.5f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1)) if len(a)>1 else '')
