import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,2300)
 if d is None or len(d)<200: d=get_index_daily_data(s,2300)
 if d is not None and len(d)>0:
  x=d[['date','close']].drop_duplicates('date').copy(); x['symbol']=s; rows.append(x)
p=pd.concat(rows,ignore_index=True); wide=p.pivot(index='date',columns='symbol',values='close').sort_index().ffill()
r=wide.pct_change()
# Trend-confirmed risk-adjusted momentum: 30-session return scaled by realized risk,
# with a continuous confirmation bonus for 10-session trend; winsorization avoids crypto dominance.
vol=r.rolling(30,min_periods=20).std()*np.sqrt(30)
raw=wide.pct_change(30)/vol
confirm=np.tanh(wide.pct_change(10)*8.0)
f=(raw*(1.0+0.35*confirm)).clip(-5,5)

def calc(h,start=None,end=None):
 fut=wide.shift(-h)/wide-1; qs=[]; ns=[]; idx=f.index
 if start: idx=idx[(idx>=start)&(idx<=end)]
 for dt in idx:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): qs.append(q); ns.append(len(z))
 q=pd.Series(qs); return len(q),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),np.mean(ns)
print('cutoff',wide.index.max().date(),'dates',len(wide),'instruments',len(wide.columns))
for h in [1,3,5,10]: print('H',h,'n IC mean std ICIR hit avgN',calc(h))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-01-27')]: print('REG',a,b,calc(1,a,b))
print('coverage',f.notna().mean().mean(),'active_dates',f.notna().any(axis=1).sum())
rank=f.rank(axis=1,pct=True); print('rank_turnover',((rank-rank.shift()).abs().mean(axis=1)).mean())
out=f.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_3_20270128_trend_confirmed_risk_momentum_signal.csv',index=False)
# max correlation with existing signal artifacts when aligned
import glob
cors=[]
for path in glob.glob('scripts/*_signal.csv'):
 try:
  q=pd.read_csv(path); q=q.pivot(index='date',columns='symbol',values='signal').reindex(f.index).reindex(columns=f.columns)
  a=f.stack(); b=q.stack(); z=pd.concat([a,b],axis=1).dropna()
  if len(z)>100: cors.append(abs(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')))
 except Exception: pass
print('max_abs_library_correlation',max(cors) if cors else None)
