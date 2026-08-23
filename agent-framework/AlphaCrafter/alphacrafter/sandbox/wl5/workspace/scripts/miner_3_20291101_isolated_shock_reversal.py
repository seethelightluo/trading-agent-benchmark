import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-10-31'); px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date'])
 px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change(); med=r.T.rolling(5).median().T; v=r.rolling(20,min_periods=15).std()
shock=(r.rolling(5).sum()-5*med).div(v.clip(lower=1e-5))
dd=p/p.rolling(60,min_periods=40).max()-1
# Isolated negative shock reversal, gated by asset drawdown and broad breadth not in panic.
breadth=(r.rolling(20).sum()>0).mean(axis=1)
mask=(shock < -0.75)&(dd<0); mask=mask.mul(breadth.gt(0.20), axis=0)
raw=(-shock).where(mask,0.)
s=raw.rank(axis=1,pct=True)
for h in [3,5,10,20]:
 vals=[]; ds=[]; ns=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: vals.append(q.f.corr(q.y,method='spearman'));ds.append(p.index[i]);ns.append(len(q))
 x=pd.Series(vals,index=ds);print('ALL',h,'dates',len(x),'IC %.6f ICIR %.6f hit %.4f coverage %.4f'%(x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(np.array(ns)/15)))
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20291101_isolated_shock_reversal_signal.csv',index=False)
print('turnover %.6f rows %d'%(s.diff().abs().mean().mean(),len(out)))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-10-31')]:
 z=[]
 for i in range(len(p)-10):
  if pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b):
   q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
   if len(q)>=8 and q.f.nunique()>1:z.append(q.f.corr(q.y,method='spearman'))
 z=pd.Series(z);print('REG10',a,'dates',len(z),'IC %.6f ICIR %.6f'%(z.mean(),z.mean()/z.std(ddof=1)))
