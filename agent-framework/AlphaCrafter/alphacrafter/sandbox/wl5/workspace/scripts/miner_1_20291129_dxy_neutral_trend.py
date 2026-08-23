import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2029-11-28')
def load(name, macro=False):
    path=Path('../persistent/index_data' if macro else '../persistent/stock_data')/(name+'.csv')
    d=pd.read_csv(path,parse_dates=['date']).sort_values('date')
    return d[d.date<=cut].set_index('date').close
p=pd.DataFrame({s:load(s) for s in U}).sort_index().dropna()
dxy=load('DXY',True).reindex(p.index).ffill()
r=p.pct_change(); v=r.rolling(20,min_periods=15).std()
# DXY-neutralized medium trend: asset 30d risk-adjusted return, penalized when its move is unusually aligned with USD strength.
mom=p.pct_change(30); dm=dxy.pct_change(30)
beta=r.rolling(60,min_periods=40).cov(dm).div(dm.rolling(60,min_periods=40).var(),axis=0)
resid=mom-beta.mul(dm,axis=0)
sig=(resid/(v*np.sqrt(30)).clip(lower=1e-6)).rank(axis=1,pct=True)
def calc(h,start=None,end=None):
 vals=[]; ns=[]; dates=[]
 for i in range(len(p)-h):
  dt=p.index[i]
  if start and not(pd.Timestamp(start)<=dt<=pd.Timestamp(end)): continue
  q=pd.concat([sig.iloc[i].rename('f'),(p.iloc[i+h]/p.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   vals.append(q.f.corr(q.y,method='spearman')); ns.append(len(q)); dates.append(dt)
 x=pd.Series(vals,index=dates).dropna()
 return len(x),float(np.mean(ns)),float(x.mean()),float(x.mean()/x.std(ddof=1)),float(np.mean(x>0)),float(np.mean(np.array(ns)/15))
print('assets',len(U),'rows',len(p),'range',p.index.min().date(),p.index.max().date())
for h in [5,10,20]: print('ALL',h,calc(h))
for a,b in [('2020-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2028-12-31'),('2029-01-01','2029-11-28')]: print('REG10',a,b,calc(10,a,b))
print('turnover',float(sig.diff().abs().mean().mean()))
out=sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna(); out.to_csv('scripts/miner_1_20291129_dxy_neutral_trend_signal.csv',index=False); print('artifact_rows',len(out),'latest',out.date.max())
