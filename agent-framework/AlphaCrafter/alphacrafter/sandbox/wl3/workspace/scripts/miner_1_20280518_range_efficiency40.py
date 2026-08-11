import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','open','close','high','low']].drop_duplicates('date'); x['symbol']=s; rows.append(x)
a=pd.concat(rows).set_index(['date','symbol']).sort_index()
# Wilder-style daily true range relative to prior close; signal is lagged 40d net return per cumulative true range.
a['prevclose']=a.groupby(level=1)['close'].shift(1)
a['tr']=pd.concat([(a.high-a.low), (a.high-a.prevclose).abs(), (a.low-a.prevclose).abs()],axis=1).max(axis=1)
cl=a['close'].unstack().sort_index().ffill(); ret=cl.pct_change()
tr=a['tr'].unstack().reindex(cl.index).ffill()/cl
net=ret.rolling(40,min_periods=30).sum(); path=tr.rolling(40,min_periods=30).sum()
f=(net/(path+1e-12)).shift(1)
fw=cl.shift(-1)/cl-1
ics={}
for h in [1,3,5,10]:
 q=[]; ns=[]
 for dt in cl.index:
  z=pd.concat([f.loc[dt],cl.shift(-h).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): q.append(v);ns.append(len(z))
 q=pd.Series(q);ics[h]=q
 print('horizon',h,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)),'hit',(q>0).mean(),'avgN',np.mean(ns))
for label,mask in [('2020-22',cl.index<'2023-01-01'),('2023-25',(cl.index>='2023-01-01')&(cl.index<'2026-01-01')),('2026+',cl.index>='2026-01-01'),('recent120',np.arange(len(cl))>=len(cl)-120)]:
 vals=[]
 for dt in cl.index[mask]:
  z=pd.concat([f.loc[dt],cl.shift(-5).loc[dt]/cl.loc[dt]-1],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v)
 q=pd.Series(vals);print('regime',label,'obs',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
valid=[pd.concat([f.loc[dt],cl.shift(-5).loc[dt]/cl.loc[dt]-1],axis=1).dropna().shape[0] for dt in cl.index]
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'dates_ge8',sum(np.array(valid)>=8),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20280518_range_efficiency40_signal.csv',index=False)
