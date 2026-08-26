import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; xs={}
for s in U:
 d=get_stock_daily_data(s,days=4000)
 if d is not None and len(d)>150:
  d=d[['date','close']].dropna().set_index('date'); r=d.close.pct_change(); xs[s]=pd.DataFrame({'f':d.close.pct_change(60)/r.rolling(20).std(),'fwd':d.close.shift(-10)/d.close-1})
all_dates=sorted(set().union(*[x.index for x in xs.values()])); ics=[]; nobs=[]; obsdates=[]
for dt in all_dates:
 z=pd.DataFrame({s:xs[s].loc[dt] for s in xs if dt in xs[s].index}).T.dropna()
 if len(z)>=8:
  ic=z.f.corr(z.fwd,method='spearman')
  if np.isfinite(ic): ics.append(ic); nobs.append(len(z)); obsdates.append(dt)
arr=np.array(ics)
def out(mask):
 a=arr[mask]; return (len(a),float(a.mean()),float(a.mean()/a.std(ddof=1)) if len(a)>1 and a.std(ddof=1)>0 else np.nan,float((a>0).mean()))
print('period',min(obsdates),max(obsdates),'dates',len(arr),'mean_n',np.mean(nobs),'coverage',np.mean(nobs)/15)
print('full',out(np.ones(len(arr),bool))); print('recent252',out(np.arange(len(arr))>=max(0,len(arr)-252))); print('online_2026+',out(np.array([d>=pd.Timestamp('2026-07-16') for d in obsdates])))
for a,b in [('2020','2023'),('2024','2026'),('2027','2029')]: print(a+'-'+b,out(np.array([(str(d)[:4]>=a and str(d)[:4]<=b) for d in obsdates])))
prev=None; ts=[]
for dt in all_dates:
 vals={s:xs[s].loc[dt,'f'] for s in xs if dt in xs[s].index}; z=pd.Series(vals).dropna()
 if len(z)>=8:
  rank=z.rank(pct=True)
  if prev is not None:
   common=rank.index.intersection(prev.index); ts.append(np.mean(np.abs(rank.loc[common]-prev.loc[common])))
  prev=rank
print('rank_turnover_proxy',float(np.mean(ts)) if ts else np.nan,'n',len(ts))
