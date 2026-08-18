import os, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr

U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date')['close'].sort_index()
 px[s]=d
P=pd.DataFrame(px).sort_index()
# shared dates and lagged data, no current-day information in signal
r20=P/P.shift(20)-1
csmean=r20.mean(axis=1)
breadth=(r20>0).mean(axis=1)
# Extreme breadth gives stronger contrarian opportunity; centered residual reversal
extreme=(2*(breadth-0.5).abs()).clip(0,1)
sig=-(r20.sub(csmean,axis=0))*(1+1.5*extreme.values[:,None])
sig=sig.shift(1)
# standardize per date for artifact and robust comparison
sig=sig.sub(sig.mean(axis=1),axis=0)
future=P.shift(-10)/P-1
rows=[]; obs=[]
for dt in P.index:
 a=sig.loc[dt]; y=future.loc[dt]
 z=pd.concat([a,y],axis=1).dropna()
 if len(z)>=8:
  ic=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  rows.append((dt,ic,len(z)))
  for s in z.index: obs.append((dt,s,float(a[s])))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print(json.dumps({'factor':'breadth_extreme_residual_reversal','horizon':10,'dates':len(q),'avg_n':round(q.n.mean(),2),'coverage':round(len(obs)/(len(P)*len(U)),4),'ic':round(q.ic.mean(),6),'icir':round(q.ic.mean()/q.ic.std()*np.sqrt(252),4),'hit':round((q.ic>0).mean(),4),'turnover':round(sig.rank(pct=True,axis=1).diff().abs().mean().mean(),4)},indent=2))
for start in ['2028-01-01','2031-01-01','2032-01-01']:
 z=q.loc[start:]; print(start, len(z), round(z.ic.mean(),6), round(z.ic.mean()/z.ic.std()*np.sqrt(252),4))
os.makedirs('scripts/artifacts',exist_ok=True)
pd.DataFrame(obs,columns=['date','symbol','signal']).to_csv('scripts/artifacts/miner_2_20330203_breadth_extreme_reversal_signal.csv',index=False)
