import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-10-31')
px={}
for s in U:
 d=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).drop_duplicates('date')
 px[s]=d.set_index('date').close
p=pd.DataFrame(px).sort_index().ffill().loc[:cut]; r=p.pct_change()
# Breakout-recovery factor: distance above trailing 60d low, normalized by recent risk,
# with a causal 5d confirmation. This favors persistent recovery/breakout assets.
low60=p.rolling(60,min_periods=45).min(); vol20=r.rolling(20,min_periods=15).std()
confirm=(p.pct_change(5)>0).astype(float)*.5+.5
raw=(p/low60-1)/(vol20*np.sqrt(20)+1e-12)*confirm
sig=raw.rank(axis=1,pct=True)
def calc(h,a=None,b=None):
 vals=[]; ns=[]; dates=[]
 for i in range(260,len(p)-h):
  dt=p.index[i]
  if a and not(pd.Timestamp(a)<=dt<=pd.Timestamp(b)): continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=q.f.corr(q.y,method='spearman')
   if np.isfinite(v): vals.append(v); ns.append(len(q)); dates.append(dt)
 x=np.asarray(vals)
 return len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)*np.sqrt(252)),float((x>0).mean()),float(np.mean(ns)/15)
print('rows',len(p),'assets',len(U),'range',p.index.min().date(),p.index.max().date(),'cut',cut.date())
for h in [5,10,20]: print('ALL',h,calc(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-10-31')]: print('REG10',a,b,calc(10,a,b))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_2_20291101_breakout_recovery_signal.csv',index=False)
print('artifact_rows',len(out),'dates',out.date.nunique(),'assets',out.symbol.nunique(),'coverage',round(sig.notna().mean().mean(),6))
print('turnover',round(sig.diff().abs().mean().mean(),6)); print('max_abs_library_correlation',None)
print('recent_mean_signal',raw.tail(1).T.sort_values(by=raw.tail(1).index[0]).tail(5).T.to_string())
