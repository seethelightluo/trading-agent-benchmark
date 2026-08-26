import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p): D[s]=pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].replace(0,np.nan)
px=pd.concat(D,axis=1).sort_index(); r=px.pct_change(); bench=r.mean(axis=1)
res10=px.pct_change(10).sub(px.pct_change(10).mean(axis=1),axis=0)
# Observation-only VIX: causal rolling percentile, high stress increases short-horizon residual snapback
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(px.index).ffill()
vp=v.rolling(252,min_periods=60).rank(pct=True)
mult=(0.65+1.35*vp).clip(.65,2.0)
factor=-res10.mul(mult,axis=0)
rows=[]
for i in range(30,len(px)-10):
 z=pd.concat([factor.iloc[i],px.iloc[i+10]/px.iloc[i]-1],axis=1).dropna()
 if len(z)>=8: rows.append((px.index[i],spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'meanN',a.n.mean(),'coverage',a.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean(),factor.rank(axis=1,pct=True).diff().abs().mean().mean()))
for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-12-31'),('2029','2031-12-31'),('2032','2035-12-20')]:
 q=a.loc[lo:hi].ic; print(lo,hi,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
for h in [5,10,20]:
 rr=[]
 for i in range(30,len(px)-h):
  z=pd.concat([factor.iloc[i],px.iloc[i+h]/px.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 print('h',h,'dates',len(rr),'IC',np.mean(rr),'ICIR',np.mean(rr)/np.std(rr,ddof=1))
out=factor.copy(); out.index.name='date'; out.to_csv('scripts/miner_1_20351220_vix_conditioned_residual_reversal_signal.csv')
