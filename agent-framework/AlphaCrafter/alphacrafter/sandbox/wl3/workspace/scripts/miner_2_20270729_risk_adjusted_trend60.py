import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,3000)
 if d is None or len(d)<200: d=get_index_daily_data(s,3000)
 if d is not None:
  x=d[['date','close']].drop_duplicates('date');x['symbol']=s;rows.append(x)
a=pd.concat(rows);p=a.pivot(index='date',columns='symbol',values='close').sort_index().ffill();r=p.pct_change()
# Risk-adjusted medium trend: 60-session return divided by trailing 40-session volatility,
# with cross-sectional median demeaning and clipping. Uses only completed prices at each date.
vol=r.rolling(40,min_periods=30).std()*np.sqrt(40)
f=(p.pct_change(60)/(vol+1e-8)).sub((p.pct_change(60)/(vol+1e-8)).median(axis=1),axis=0).clip(-6,6)
print('cutoff',p.index.max().date(),'dates',len(p),'instruments',len(p.columns))
def calc(h):
 fut=p.shift(-h)/p-1;qs=[];ns=[];ds=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fut.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):qs.append(q);ns.append(len(z));ds.append(dt)
 q=pd.Series(qs,index=pd.DatetimeIndex(ds));return len(q),np.mean(ns),q.mean(),q.std(ddof=1),q.mean()/q.std(ddof=1)*np.sqrt(len(q)),(q>0).mean(),q
for h in [1,3,5,10]:
 z=calc(h);print('H',h,'obs',z[0],'avgN',round(z[1],2),'IC',round(z[2],6),'ICIR',round(z[4],6),'hit',round(z[5],4))
z=calc(1)[-1]
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01',str(p.index.max().date()))]:
 q=z.loc[a:b];print('REG',a,b,'obs',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1)*np.sqrt(len(q)),4))
print('coverage',round(float(f.notna().mean().mean()),6),'turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),6))
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_2_20270729_risk_adjusted_trend60_signal.csv',index=False)
print('artifact scripts/miner_2_20270729_risk_adjusted_trend60_signal.csv')
print('max_abs_library_correlation',None)
