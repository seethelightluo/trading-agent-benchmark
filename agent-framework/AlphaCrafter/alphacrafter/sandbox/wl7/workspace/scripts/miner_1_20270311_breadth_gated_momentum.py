import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; CUT=pd.Timestamp('2027-03-10')
def load(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,3000)
   if x is not None and len(x):
    x=x.copy();x.date=pd.to_datetime(x.date).dt.normalize();return x.drop_duplicates('date').set_index('date').sort_index()
  except Exception: pass
D={s:load(s) for s in U};D={s:x for s,x in D.items() if x is not None}
P=pd.concat([D[s]['close'].rename(s) for s in D],axis=1).sort_index().loc[:CUT]; P=P.apply(pd.to_numeric,errors='coerce'); r=P.pct_change(); mom=P.pct_change(30); vol=r.rolling(30,min_periods=20).std(); breadth=(r.rolling(20).mean()>0).mean(axis=1); F=(mom/vol*(.5+breadth)).shift(1)
print('data',len(D),len(P),F.notna().mean().mean())
def metrics(FR,lo=None,hi=None):
 vals=[];ns=[]
 for dt in F.loc[lo:hi].index:
  q=pd.concat([F.loc[dt],FR.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1: vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'));ns.append(len(q))
 a=pd.Series(vals);return len(a),np.mean(ns),a.mean(),a.mean()/a.std(ddof=1)*np.sqrt(252),(a>0).mean()
for h in [1,5,10,20]:
 z=metrics(P.shift(-h)/P-1);print('horizon',h,'dates',z[0],'avg_assets',round(z[1],2),'IC',round(z[2],5),'ICIR',round(z[3],4),'hit',round(z[4],4))
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-10'),('2026-07-16','2027-03-10')]: print('regime',lo,hi,metrics(P.shift(-1)/P-1,lo,hi))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean());F.rank(axis=1,pct=True).to_csv('scripts/miner_1_20270311_breadth_gated_momentum_signal.csv')
