import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2029-09-05');px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna();r=p.pct_change(); neg=r.clip(upper=0); down=np.sqrt((neg**2).rolling(30,min_periods=15).mean());mom=p.pct_change(60);sig=(mom/down.clip(lower=1e-5)).rank(axis=1,pct=True)
def calc(h,a=None,b=None):
 z=[];ns=[];ds=[]
 for i in range(len(p)-h):
  dt=p.index[i]
  if a and not(pd.Timestamp(a)<=dt<=pd.Timestamp(b)):continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:z.append(q.f.corr(q.y,method='spearman'));ns.append(len(q));ds.append(dt)
 x=pd.Series(z,index=ds);return len(x),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0),np.mean(np.array(ns)/15)
print('rows',len(p),'assets',len(U))
for h in [5,10,20]:print('ALL',h,calc(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-09-05')]:print('REG10',a,b,calc(10,a,b))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20290906_downside_momentum_signal.csv',index=False);print('artifact_rows',len(out),'latest',out.date.max())
