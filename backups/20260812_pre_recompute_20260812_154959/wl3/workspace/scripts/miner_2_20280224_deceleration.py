import numpy as np,pandas as pd,glob,os
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200:d=get_index_daily_data(s,3000)
 if d is not None and len(d):
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
cl=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index().ffill();r=cl.pct_change();vol=r.rolling(20,min_periods=15).std()
# Positive signal means recent trend has weakened versus preceding trend: contrarian deceleration.
f=(cl.pct_change(20).shift(10)-cl.pct_change(10)).shift(1)/vol.shift(1).replace(0,np.nan)
qs=[];ns=[]
for dt in cl.index:
 z=pd.concat([f.loc[dt],(cl.shift(-5)/cl-1).loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):qs.append(q);ns.append(len(z))
q=pd.Series(qs); ic=q.mean(); icir=ic/q.std(ddof=1)*np.sqrt(len(q))
print('cutoff',cl.index.max().date(),'dates',len(cl),'instruments',len(cl.columns),'obs',len(q),'avgN',np.mean(ns),'IC',ic,'ICIR',icir,'hit',(q>0).mean(),'coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [1,3,5,10]:
 qq=[];fu=cl.shift(-h)/cl-1
 for dt in cl.index:
  z=pd.concat([f.loc[dt],fu.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(v):qq.append(v)
 print('h',h,'IC',np.mean(qq),'n',len(qq))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20280224_deceleration_signal.csv',index=False)
# Artifact correlations, same date-symbol intersection, robustly cap scan.
mx=0.;who='none';a=out.set_index(['date','symbol']).signal
for p in glob.glob('scripts/*_signal.csv'):
 if p.endswith('deceleration_signal.csv'):continue
 try:
  b=pd.read_csv(p,parse_dates=['date']).set_index(['date','symbol']).iloc[:,0]
  z=pd.concat([a,b],axis=1).dropna()
  if len(z)>100:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c) and abs(c)>mx:mx=abs(c);who=os.path.basename(p)
 except Exception:pass
print('max_abs_library_correlation',mx,'artifact',who)
