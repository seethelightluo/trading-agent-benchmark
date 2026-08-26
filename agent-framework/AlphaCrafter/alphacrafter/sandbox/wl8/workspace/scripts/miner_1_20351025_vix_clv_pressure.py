import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
pxs={}; highs={}; lows={}
for s in U:
 d=get_stock_daily_data(s,6000)
 if d is None or len(d)<100: d=get_index_daily_data(s,6000)
 if d is not None:
  d=d.drop_duplicates('date').set_index('date'); pxs[s]=d.close; highs[s]=d.high; lows[s]=d.low
px=pd.DataFrame(pxs).sort_index().ffill(); hi=pd.DataFrame(highs).reindex(px.index).ffill(); lo=pd.DataFrame(lows).reindex(px.index).ffill()
clv=((px-lo)/(hi-lo).replace(0,np.nan)-0.5); pressure=clv.rolling(10,min_periods=7).mean()
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']); v=v.drop_duplicates('date').set_index('date'); vix=v.close.reindex(px.index).ffill()
vz=((vix-vix.rolling(120,min_periods=60).median())/vix.rolling(120,min_periods=60).std()).clip(-2,2)
sig=(-pressure.mul(1+0.35*vz,axis=0)).shift(1)
ics=[]; ns=[]; turns=[]; prev=None
for dt in sig.index:
 y=(px.shift(-10)/px-1).loc[dt]; z=pd.concat([sig.loc[dt],y],axis=1).dropna()
 if len(z)>=8:
  c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(c): ics.append((dt,c));ns.append(len(z))
 x=sig.loc[dt].rank(pct=True).dropna()
 if prev is not None:
  q=x.index.intersection(prev.index)
  if len(q): turns.append(abs(x[q]-prev[q]).mean())
 prev=x
ic=pd.Series(dict(ics)); print('dates',len(ic),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15,'IC',ic.mean(),'ICIR_daily_paper',ic.mean()/ic.std()*np.sqrt(252),'hit',np.mean(ic>0),'turnover',np.mean(turns))
for h in [1,5,20]:
 a=[]; y=px.shift(-h)/px-1
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): a.append(c)
 print('decay',h,np.mean(a),len(a))
for n in [365,750,1260]:
 q=ic.tail(n); print('recent',n,q.mean(),q.mean()/q.std()*np.sqrt(252),len(q))
print('range',ic.index.min(),ic.index.max())
sig.reset_index().melt(id_vars='date',var_name='symbol',value_name='signal').dropna().to_csv('scripts/miner_1_20351025_vix_clv_pressure_signal.csv',index=False)
pd.DataFrame({'date':ic.index,'ic':ic.values}).to_csv('scripts/miner_1_20351025_vix_clv_pressure_ic.csv',index=False)
