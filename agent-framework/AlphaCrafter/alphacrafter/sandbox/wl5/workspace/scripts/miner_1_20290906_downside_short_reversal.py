import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2029-09-05');p={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);p[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(p).sort_index().dropna();r=p.pct_change();v=r.rolling(20,min_periods=15).std();down=r.clip(upper=0).abs().rolling(20,min_periods=15).mean();sig=(-r.rolling(3).sum()/down.clip(lower=1e-5)).rank(axis=1,pct=True)
def calc(h,a,b):
 z=[];n=[];ds=[]
 for i in range(len(p)-h):
  if not(pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b)):continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8:z.append(q.f.corr(q.y,method='spearman'));n.append(len(q));ds.append(p.index[i])
 x=pd.Series(z,index=ds).dropna();return len(x),np.mean(n),x.mean(),x.mean()/x.std(ddof=1),np.mean(x>0)
print('rows',len(p),'assets',p.shape[1])
for h in [5,10,20]:print('ALL',h,calc(h,'2020-01-01','2029-09-05'))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-09-05')]:print('REG10',a,b,calc(10,a,b))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_1_20290906_downside_short_reversal_signal.csv',index=False);print('artifact',len(out))
