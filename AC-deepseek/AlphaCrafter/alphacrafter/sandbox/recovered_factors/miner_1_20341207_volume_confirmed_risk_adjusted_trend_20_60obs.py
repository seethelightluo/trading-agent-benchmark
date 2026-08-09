"""Single candidate: volume-confirmed risk-adjusted trend (20/60 sessions)."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; raw={}
for a in assets:
    d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
    raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a:raw[a]['close'].astype(float) for a in assets},axis=1).sort_index()
volume=pd.concat({a:raw[a]['volume'].astype(float) if 'volume' in raw[a] else pd.Series(dtype=float) for a in assets},axis=1).reindex(close.index)
r=close.pct_change()
# A persistent price trend is more credible where its recent traded-volume participation is above its own baseline.
trend=close.pct_change(20).div(r.rolling(20,min_periods=15).std().replace(0,np.nan))
dvol=(close*volume).where((close>0)&(volume>0))
participation=dvol.rolling(5,min_periods=4).median().div(dvol.rolling(60,min_periods=45).median().replace(0,np.nan))
# logarithm limits a few very liquid benchmark contracts dominating the signal.
f=trend*np.log(participation).clip(-2,2)
f=f.replace([np.inf,-np.inf],np.nan)
def ev(h):
 fw=close.shift(-h).div(close).sub(1); vals=[]; breadth=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('f'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.y.nunique()>1:
   vals.append((dt,z.f.corr(z.y,method='spearman'))); breadth.append(len(z))
 x=pd.Series(dict(vals),dtype=float); sd=x.std(ddof=1)
 return x,dict(dates=len(x),ic=float(x.mean()),icir=float(x.mean()/sd),hit_ratio=float((x>0).mean()),mean_instruments=float(np.mean(breadth)),min_instruments=int(min(breadth)))
print('FACTOR volume_confirmed_risk_adjusted_trend_20_60obs')
print('expression=(close/close_lag20-1)/std_20(return)*clip(log(median_5(close*volume)/median_60(close*volume)),-2,2)')
print('visible_history',close.index.min().date(),close.index.max().date(),'assets',len(assets),'volume_nonnull',volume.notna().mean().to_dict())
for h in (1,5,10,20):
 x,s=ev(h);print('HORIZON',h,s)
 for name,mask in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
  y=x[mask]; print('REGIME',h,name,'dates',len(y),'ic',float(y.mean()),'icir',float(y.mean()/y.std(ddof=1)) if len(y)>1 else np.nan,'hit',float((y>0).mean()))
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 z=ranks.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('COVERAGE',float(f.notna().mean().mean()),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'turnover',float(np.mean(turns)),'turn_dates',len(turns),'median_iqr',float(iqr.median()))
print('Admission needs full-sample same-horizon |IC|>=.007, |ICIR|>=.084 and max library signal rho<.5.')
