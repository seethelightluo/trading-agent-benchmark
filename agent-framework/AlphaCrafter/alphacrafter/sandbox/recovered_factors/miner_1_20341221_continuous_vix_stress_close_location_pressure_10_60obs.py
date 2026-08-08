"""Single idea: continuous VIX-stress-conditioned close-location pressure.
A close near its daily high/lows carries more cross-asset information when the prior
VIX level is high relative to its own 60-day history.  The continuous lagged VIX
multiplier avoids a sparse binary state; cross-sectional 20-day trend is removed.
All inputs at t are completed or lagged observations; forward returns are evaluation only.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; raw={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
 raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
def field(x): return pd.concat({a:pd.to_numeric(raw[a][x],errors='coerce') for a in A},axis=1).sort_index()
c,h,l=field('close'),field('high'),field('low'); r=c.pct_change(fill_method=None)
v=get_index_daily_data('VIX',5000).copy();v['date']=pd.to_datetime(v.date);v=v.drop_duplicates('date').set_index('date').sort_index()
vc=pd.to_numeric(v.close,errors='coerce').reindex(c.index).ffill()
# strictly lagged, clipped standardized VIX state; 0 denotes normal stress.
vz=(vc.shift(1)-vc.rolling(60,min_periods=45).mean().shift(1)).div(vc.rolling(60,min_periods=45).std().shift(1).clip(lower=1e-8)).clip(-2,2)
clv=(c-l).div((h-l).replace(0,np.nan)).sub(.5).clip(-.5,.5)
# 10 completed-bar average, amplifying close-location only under relatively high stress.
rawf=clv.mul(1+vz.clip(lower=0),axis=0).rolling(10,min_periods=7).mean()
trend=c.div(c.shift(20)).sub(1).div(r.rolling(20,min_periods=15).std().clip(lower=1e-8))
f=pd.DataFrame(np.nan,index=c.index,columns=A)
for dt in c.index:
 q=pd.concat([rawf.loc[dt].rename('x'),trend.loc[dt].rename('t')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8 and q.x.nunique()>1 and q.t.nunique()>1:
  b=np.polyfit(q.t,q.x,1);f.loc[dt,q.index]=q.x-(b[0]*q.t+b[1])
def ev(hz):
 fw=c.shift(-hz).div(c).sub(1);z=[];ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.x.nunique()>1 and q.y.nunique()>1:z.append((dt,q.x.corr(q.y,method='spearman')));ns.append(len(q))
 return pd.Series(dict(z)),ns
print('FACTOR continuous_vix_stress_close_location_pressure_10_60obs')
print('EXPRESSION cs_residual(mean_10(((close-low)/(high-low)-0.5)*(1+max(zscore_60(VIX[t-1]),0))),risk_adjusted_trend_20)')
print('endpoint',c.index.max().date(),'assets',len(A),'vix_nonmissing',int(vc.notna().sum()))
for hz in (1,5,10,20):
 x,ns=ev(hz); ir=x.mean()/x.std(ddof=1)
 print('HORIZON',hz,'IC',round(x.mean(),6),'ICIR',round(ir,6),'hit',round((x>0).mean(),6),'dates',len(x),'mean_n',round(np.mean(ns),3),'min_n',min(ns),'PASS',abs(x.mean())>=.007 and abs(ir)>=.084)
 for nm,mask in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
  y=x[mask];print('REGIME',hz,nm,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),6))
rk=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('QUALITY coverage',round(f.notna().mean().mean(),6),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'turnover',round(np.mean(turn),6),'turnover_comparisons',len(turn),'median_iqr',round(iqr.median(),8))
f.to_pickle('scripts/miner_1_20341221_continuous_vix_stress_close_location_pressure_10_60obs_signal.pkl')
