import pandas as pd, numpy as np, glob, os
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in assets:
 f='../persistent/stock_data/'+a+'.csv'
 x=pd.read_csv(f,parse_dates=['date']).sort_values('date').set_index('date')
 D[a]=x.close.astype(float)
p=pd.DataFrame(D).sort_index()
# factor: continuation return normalized by peak-to-trough drawdown over trailing 60 sessions;
# reward persistent positive returns while penalizing fragile/high-drawdown assets.
ret20=p/p.shift(20)-1
rollmax=p.shift(1).rolling(60,min_periods=40).max()
dd=(p.shift(1)/rollmax-1).abs()
maxdd=dd.rolling(60,min_periods=40).max()
f=ret20/(0.02+maxdd)
# cap extreme values cross-sectionally only for robustness
f=f.clip(-10,10)
for h in [5,10,20]:
 fr=p.shift(-h)/p-1
 ics=[]; ns=[]; turnovers=[]
 prev=None
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  ranks=f.loc[dt].rank(pct=True)
  if prev is not None:
   q=pd.concat([ranks,prev],axis=1).dropna(); turnovers.append(np.mean(np.abs(q.iloc[:,0]-q.iloc[:,1])))
  prev=ranks
 ic=np.array(ics); print({'horizon':h,'dates':len(ic),'avg_n':np.mean(ns),'ic':np.mean(ic),'icir':np.mean(ic)/np.std(ic,ddof=1),'hit':np.mean(ic>0),'turnover':np.nanmean(turnovers)})
 # regime breakdown for 20d
 if h==20:
  # reconstruct dates for regime
  vals=[]
  for dt in p.index:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
  q=pd.DataFrame(vals,columns=['date','ic']).set_index('date'); print(q.groupby(q.index.year).ic.agg(['mean','count']).to_string())
print('data_end',p.index.max().date(),'factor_valid',f.notna().sum(axis=1).mean())
