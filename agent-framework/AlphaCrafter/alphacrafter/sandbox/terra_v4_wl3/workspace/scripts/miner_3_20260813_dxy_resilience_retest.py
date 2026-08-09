import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2026-07-15')
def load(s):
 f='../persistent/stock_data/'+s+'.csv'; d=pd.read_csv(f,parse_dates=['date']); d=d.drop_duplicates('date').set_index('date').sort_index(); return d['close'].loc[:end]
def macro():
 d=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return d.close.loc[:end]
p=pd.concat({s:load(s) for s in U},axis=1,join='outer').sort_index(); d=macro()
# Returns and exact intersection: no forward fill, only completed observations.
r=p.pct_change(); dr=d.pct_change(); common=r.index.intersection(dr.index); r=r.loc[common]; dr=dr.loc[common]; p=p.loc[common]
F=pd.DataFrame(index=common,columns=U,dtype=float); w=40
for i,dt in enumerate(common):
 if i<w-1: continue
 x=dr.iloc[i-w+1:i+1]
 for s in U:
  z=pd.concat([x,r[s].iloc[i-w+1:i+1]],axis=1).dropna()
  down=z[z.iloc[:,0]<0]
  if len(down)>=12 and down.iloc[:,0].var()>1e-12: F.loc[dt,s]=-down.iloc[:,0].cov(down.iloc[:,1])/down.iloc[:,0].var()
# signal artifact for reproducible provenance
out='scripts/miner_3_20260813_dxy_resilience_signal.csv'; F.to_csv(out)
A=[]; dates=[]; ns=[]
for i,dt in enumerate(common[:-1]):
 z=pd.concat([F.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): A.append(q);dates.append(dt);ns.append(len(z))
a=np.array(A); print('dates',len(a),'range',dates[0],dates[-1],'avgN',np.mean(ns),'coverage',F.notna().mean().mean(),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
for h in [5,10]:
 vals=[]
 for dt in common[:-h]:
  z=pd.concat([F.loc[dt],(p.shift(-h)/p-1).loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 vals=np.array(vals); print('h',h,'n',len(vals),'IC',vals.mean(),'ICIR',vals.mean()/vals.std(ddof=1))
for lo,hi in [('2020','2022-12-31'),('2023','2024-12-31'),('2025','2026-07-15')]:
 q=a[(pd.DatetimeIndex(dates)>=pd.Timestamp(lo))&(pd.DatetimeIndex(dates)<=pd.Timestamp(hi))]; print('regime',lo,len(q),q.mean() if len(q) else np.nan,q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
# pooled correlations with common basic library proxies
for n,x in {'mom20':r.rolling(20).sum(),'rev5':-r.rolling(5).sum(),'clv':p/p.rolling(20).max()-1}.items(): print('corr',n,F.stack().corr(x.stack()))
