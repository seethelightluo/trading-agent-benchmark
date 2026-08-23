import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2029-10-17')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill()
# stress gate: recovery after a 60d drawdown is activated only when VIX is above its trailing percentile.
dd=p/p.rolling(60,min_periods=40).max()-1; rebound=r.rolling(5).sum()
base=(rebound.clip(lower=0)*(-dd).clip(lower=0)).where(dd<0,0.)
def run(th,h):
 gate=(v>v.rolling(120,min_periods=60).quantile(th)).astype(float)
 s=(base.mul(gate,axis=0)).rank(axis=1,pct=True)
 xs=[]; ns=[]; ds=[]
 for i in range(len(p)-h):
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: xs.append(q.f.corr(q.y,method='spearman'));ns.append(len(q));ds.append(p.index[i])
 x=pd.Series(xs,index=ds);return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(np.array(ns)/15),s
print('rows',len(p),'assets',len(U),'cut',cut.date())
for th in [.5,.7,.8,.9]:
 for h in [5,10,20]:
  z=run(th,h);print('TEST',th,h,'dates',z[0],'IC',round(z[1],5),'ICIR',round(z[2],5),'hit',round(z[3],4),'coverage',round(z[4],4))
th=.8;h=10;z=run(th,h);s=z[-1]
out=s.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20291018_macro_stress_recovery_signal.csv',index=False)
print('artifact_rows',len(out),'turnover',s.rank(axis=1,pct=True).diff().abs().mean().mean(),'coverage',s.notna().mean().mean())
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-10-17')]:
 # restricted calculation
 xs=[]
 for i in range(len(p)-10):
  if not(pd.Timestamp(a)<=p.index[i]<=pd.Timestamp(b)):continue
  q=pd.concat([s.iloc[i].rename('f'),(p.iloc[i+10]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:xs.append(q.f.corr(q.y,method='spearman'))
 q=pd.Series(xs);print('REG10',a,b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
print('max_abs_library_correlation',None)
