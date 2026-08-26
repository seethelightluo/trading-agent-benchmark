import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close'].sort_index()
P=pd.concat(px,axis=1).sort_index(); R=P.pct_change()
# Inputs end strictly at t-1: signal indexed t uses prices through t-1
ret20=P.shift(1)/P.shift(21)-1
vol20=R.shift(1).rolling(20,min_periods=15).std()*np.sqrt(20)
f=ret20/vol20
# cross-sectional ranks are not needed for IC; forward returns
out=[]
for h in [10,20,40,60]:
 vals=[]
 for i in range(len(P)-h):
  # signal at date i, forward return from i close to i+h close (signal actually lagged)
  y=P.iloc[i+h]/P.iloc[i]-1; x=f.iloc[i]
  z=pd.concat([x.rename('x'),y.rename('y')],axis=1).dropna()
  if len(z)>=8:
   vals.append((z.index[0] if False else P.index[i],spearmanr(z.x,z.y).statistic,len(z)))
 a=np.array([v[1] for v in vals]);
 print('H',h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0),len(a),np.mean([v[2] for v in vals])))
# coverage and turnover on valid daily cross section
valid=f.notna().sum(axis=1)>=8
coverage=f.notna().sum().sum()/(len(f)*len(U));
ranks=f.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).where(valid).mean()
print('coverage %.4f turnover %.6f rows %d instruments %d range %s %s'%(coverage,turnover,len(P),len(U),P.index.min().date(),P.index.max().date()))
