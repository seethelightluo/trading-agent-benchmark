import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-10-17'); px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date'])
 px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().ffill(); r=p.pct_change()
# Candidate: medium-horizon trend persistence, risk-adjusted and confirmed by a positive short-term move.
# All inputs are known at date t; forward return begins after t.
ret20=p.pct_change(20); vol40=r.rolling(40,min_periods=30).std(); ret5=p.pct_change(5)
sig=(ret20/(vol40*np.sqrt(20)+1e-12))*((ret5>0).astype(float)*0.5+0.5)
sig=sig.rank(axis=1,pct=True)
def calc(h,a=None,b=None):
 vals=[]; ns=[]; dates=[]
 for i in range(260,len(p)-h):
  dt=p.index[i]
  if a and not(pd.Timestamp(a)<=dt<=pd.Timestamp(b)): continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   vals.append(q.f.corr(q.y,method='spearman')); ns.append(len(q)); dates.append(dt)
 x=pd.Series(vals,index=dates)
 return len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)*np.sqrt(252)),float((x>0).mean()),float(np.mean(np.array(ns)/15))
print('rows',len(p),'assets',len(U),'cut',cut.date())
for h in [5,10,20]: print('ALL',h,calc(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-10-17')]: print('REG10',a,b,calc(10,a,b))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20291018_trend_persistence_signal.csv',index=False)
print('artifact_rows',len(out),'dates',out.date.nunique(),'assets',out.symbol.nunique(),'coverage',sig.notna().mean().mean())
print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
print('max_abs_library_correlation',None)
