"""Single idea: inverse directional intraday efficiency persistence (20 observations)."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; rows={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date); rows[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a:rows[a].close.astype(float) for a in assets},axis=1).sort_index()
f=pd.DataFrame(index=close.index,columns=assets,dtype=float)
for a in assets:
 d=rows[a].reindex(close.index); rng=(d.high.astype(float)-d.low.astype(float)).replace(0,np.nan)
 f[a]=-((d.close.astype(float)-d.open.astype(float))/rng).rolling(20,min_periods=15).mean()
def ev(h):
 fw=close.shift(-h).div(close).sub(1); out=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: out.append((dt,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 x=pd.Series(dict(out)); return x,dict(dates=len(x),ic=x.mean(),icir=x.mean()/x.std(ddof=1),hit=(x>0).mean(),mean_n=np.mean(ns),min_n=min(ns))
print('FACTOR inverse_directional_intraday_efficiency_persistence_20obs = -mean_20((close-open)/(high-low))')
print('history',close.index.min().date(),close.index.max().date(),'universe',len(assets))
for h in (1,5,10,20):
 x,m=ev(h);print('H',h,m)
 if h==20:
  for name,mask in [('2026_2028',x.index<'2029-01-01'),('2029_2030',(x.index>='2029-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01')]:
   y=x[mask];print('REGIME',name,'dates',len(y),'IC',y.mean(),'ICIR',y.mean()/y.std(ddof=1),'hit',(y>0).mean())
r=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(r)):
 z=r.iloc[[i-1,i]].T.dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('coverage',int(f.notna().sum().sum()),'/',f.size,'=',f.notna().mean().mean(),'turnover',np.mean(turns),'median_iqr',f.quantile(.75,axis=1).sub(f.quantile(.25,axis=1)).median())
print('Admission not evaluated: complete exact library-signal Spearman audit required if IC gates pass.')
