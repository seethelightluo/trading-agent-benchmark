import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2029-11-14');px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).sort_values('date');px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna();r=p.pct_change();v=r.rolling(20,min_periods=15).std(); shock=r.rolling(5).sum()/v.clip(lower=1e-6); dd=p/p.rolling(60,min_periods=40).max()-1
# Continuation only for isolated strong moves, requiring non-panic cross-sectional breadth.
breadth=(r.rolling(20).sum()>0).mean(axis=1); med=shock.median(axis=1); isolated=shock.sub(med,axis=0).abs()>0.75
sig=(shock*isolated).where(breadth>0.2,0.).rank(axis=1,pct=True)
def calc(h,a=None,b=None):
 x=[];n=[];ds=[]
 for i in range(len(p)-h):
  if a and not(pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b)):continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:x.append(q.f.corr(q.y,method='spearman'));n.append(len(q));ds.append(p.index[i])
 x=pd.Series(x,index=ds).dropna();return len(x),np.mean(n),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0),np.mean(np.array(n)/15)
print('assets',len(U),'rows',len(p))
for h in [3,5,10,20]:print('ALL',h,calc(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-11-14')]:print('REG10',a,b,calc(10,a,b))
print('turnover',sig.diff().abs().mean().mean());out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20291115_shock_continuation_signal.csv',index=False);print('artifact_rows',len(out))
