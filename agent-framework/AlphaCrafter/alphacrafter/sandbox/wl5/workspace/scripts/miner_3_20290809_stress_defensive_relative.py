import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2029-08-08'); p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date'); p[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(p).sort_index().dropna(how='all'); p=p.dropna() # common dates ensure causal cross-section
r=p.pct_change(); defensive=p[['XAU','US10Y','CN10Y']].pct_change(30).mean(axis=1)
vol=r.rolling(20,min_periods=15).std().clip(lower=1e-5); base=-(p.pct_change(30).sub(defensive,axis=0))/(vol*np.sqrt(252))
disp=r.rolling(20,min_periods=15).std().mean(axis=1); gate=(disp>disp.rolling(120,min_periods=60).median()).astype(float)
sig=(base*(1+.30*gate)).replace([np.inf,-np.inf],np.nan).rank(axis=1,pct=True)
def calc(h,start=None,end=None):
 vals=[]; ns=[]; dates=[]; turns=[]
 for i in range(len(p)-h):
  dt=p.index[i]
  if start and not(pd.Timestamp(start)<=dt<=pd.Timestamp(end)): continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   vals.append(q.f.corr(q.y,method='spearman')); ns.append(len(q)); dates.append(dt)
   if i: turns.append(sig.iloc[i].sub(sig.iloc[i-1]).abs().mean())
 x=pd.Series(vals,index=dates).dropna(); return len(x),np.mean(ns),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0),np.mean(np.array(ns)/15),np.mean(turns)
print('assets',len(U),'rows',len(p),'range',p.index.min().date(),p.index.max().date())
for h in [5,10,20]: print('ALL',h,calc(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-08-01','2029-08-08')]: print('REG',a,b,calc(10,a,b))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_3_20290809_stress_defensive_relative_signal.csv',index=False); print('artifact_rows',len(out),'latest',out.date.max())
