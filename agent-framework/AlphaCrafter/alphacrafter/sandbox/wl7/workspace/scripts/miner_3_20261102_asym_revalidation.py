import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,3000)
   if d is not None and len(d):
    d=d.copy(); d.date=pd.to_datetime(d.date).dt.normalize(); return d.drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:fetch(s) for s in U}; D={s:d.loc[:'2026-10-30'] for s,d in D.items() if d is not None}
# Volatility-scaled asymmetric 2-day reversal. Penalize recent losses more when
# downside semivolatility is elevated; all rolling inputs are lagged at signal use.
parts=[]
for s,d in D.items():
 x=d[['close']].replace([np.inf,-np.inf],np.nan).dropna(); ret=x.close.pct_change(); vol=ret.rolling(20,min_periods=10).std(); down=ret.where(ret<0,0).rolling(20,min_periods=10).std(); dz=(down-down.rolling(60,min_periods=20).median())/(down.rolling(60,min_periods=20).std()+1e-8)
 f=(-x.close.pct_change(2)/(vol+1e-8))*(1+dz.clip(0,2)*0.5); fr=x.close.shift(-1)/x.close-1
 parts.append(pd.DataFrame({'f':f,'fr':fr},index=x.index).dropna().reset_index())
R=pd.concat(parts); print('assets',len(D),'range',R.date.min(),R.date.max())
def calc(h):
 q=[]
 for s,d in D.items():
  x=d[['close']].replace([np.inf,-np.inf],np.nan).dropna(); ret=x.close.pct_change(); vol=ret.rolling(20,min_periods=10).std(); down=ret.where(ret<0,0).rolling(20,min_periods=10).std(); dz=(down-down.rolling(60,min_periods=20).median())/(down.rolling(60,min_periods=20).std()+1e-8); f=(-x.close.pct_change(2)/(vol+1e-8))*(1+dz.clip(0,2)*0.5); fr=x.close.shift(-h)/x.close-1; q.append(pd.DataFrame({'f':f,'fr':fr}).dropna().reset_index())
 q=pd.concat(q); vals=[]; ns=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1: vals.append(g.f.corr(g.fr,method='spearman')); ns.append(len(g))
 a=pd.Series(vals); return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
for h in [1,5,10,20]: print('horizon',h,'dates avg_names IC ICIR hit',calc(h))
for y in range(2020,2027):
 q=R[R.date.dt.year==y]; a=[]
 for _,g in q.groupby('date'):
  if len(g)>=8 and g.f.nunique()>1 and g.fr.nunique()>1:a.append(g.f.corr(g.fr,method='spearman'))
 a=pd.Series(a); print('regime',y,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252) if len(a)>1 else np.nan)
# rank turnover, using cross-sectional ranks on consecutive common dates
P=R.pivot(index='date',columns=R.columns[1],values='f') if False else None
print('coverage_dates',R.date.nunique(), 'valid_assets',len(D))
