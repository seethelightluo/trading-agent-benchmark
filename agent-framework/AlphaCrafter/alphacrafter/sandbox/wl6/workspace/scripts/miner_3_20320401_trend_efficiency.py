import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).sort_values('date').set_index('date')
 D[a]=x.close.astype(float)
p=pd.DataFrame(D).sort_index(); p=p.loc[:'2032-03-31']
# Trend efficiency: signed 20-session displacement divided by path length,
# with a 60-session trend anchor; interpretable continuation signal.
r20=p/p.shift(20)-1
path=p.pct_change().abs().rolling(20,min_periods=18).sum()
anchor=p/p.shift(60)-1
f=(r20/(0.01+path))*(1+anchor.clip(-.5,.5))
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; ics=[]; ns=[]; turns=[]; prev=None
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  r=f.loc[dt].rank(pct=True)
  if prev is not None:
   q=pd.concat([r,prev],axis=1).dropna(); turns.append(np.mean(abs(q.iloc[:,0]-q.iloc[:,1])))
  prev=r
 ic=np.array(ics); print({'horizon':h,'valid_dates':len(ic),'avg_instruments':round(np.mean(ns),3),'ic':round(float(np.mean(ic)),6),'icir':round(float(np.mean(ic)/np.std(ic,ddof=1)),6),'hit_ratio':round(float(np.mean(ic>0)),4),'turnover':round(float(np.mean(turns)),6)})
 if h==20:
  vals=[]
  for dt in p.index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
  q=pd.DataFrame(vals,columns=['date','ic']).set_index('date'); print(q.groupby(q.index.year).ic.agg(['mean','count']).to_string())
print('cutoff',p.index.max().date(),'coverage',float(f.notna().sum(axis=1).mean()/15))
