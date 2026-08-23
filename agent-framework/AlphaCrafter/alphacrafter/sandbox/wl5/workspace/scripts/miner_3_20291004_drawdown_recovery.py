import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];cut=pd.Timestamp('2029-10-03');px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']);px[s]=d[d.date<=cut].set_index('date').close
p=pd.DataFrame(px).sort_index().dropna(); r=p.pct_change()
# Drawdown-recovery: recent rebound strength, scaled by the magnitude of the prior 60d drawdown.
dd=p/p.rolling(60,min_periods=40).max()-1
rebound=r.rolling(5).sum()
sig=(rebound.clip(lower=0)*(-dd).clip(lower=0)).where(dd<0,0.0)
sig=sig.rank(axis=1,pct=True)
def calc(h,a=None,b=None):
 x=[];ns=[];dates=[]
 for i in range(len(p)-h):
  dt=p.index[i]
  if a and not(pd.Timestamp(a)<=dt<=pd.Timestamp(b)):continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:x.append(q.f.corr(q.y,method='spearman'));ns.append(len(q));dates.append(dt)
 x=pd.Series(x,index=dates);return len(x),x.mean(),x.mean()/x.std(ddof=1),float((x>0).mean()),float(np.mean(np.array(ns)/15))
print('rows',len(p),'assets',len(U),'cut',cut.date())
for h in [3,5,10,20]:print('ALL',h,calc(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2028-09-01','2029-10-03')]:print('REG10',a,b,calc(10,a,b))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna();out.to_csv('scripts/miner_3_20291004_drawdown_recovery_signal.csv',index=False);print('artifact_rows',len(out),'latest',out.date.max())
rr=sig.rank(axis=1,pct=True);print('turnover',rr.diff().abs().mean().mean(),'coverage',sig.notna().mean().mean())
# yearly 10d diagnostics
for y,g in pd.Series([calc(10,str(yy)+'-01-01',str(yy)+'-12-31')[1] for yy in range(2020,2030)]).items(): print('year',2020+y,g)
# library provenance not computed here
print('max_abs_library_correlation',None)
