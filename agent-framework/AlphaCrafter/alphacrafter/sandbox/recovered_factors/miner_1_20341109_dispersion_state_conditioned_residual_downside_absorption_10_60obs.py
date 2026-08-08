"""Single idea: dispersion-state-conditioned residual downside absorption.
When cross-asset return dispersion is elevated (lagged 20d median state), measure each
asset's mean idiosyncratic repair after its prior idiosyncratic loss over 10 sessions.
The cross-sectional signal is residualized against 20d risk-adjusted trend. All inputs
at t use data no later than t; forward returns are evaluation only."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; raw={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
 raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
c=pd.concat({a:raw[a].close.astype(float) for a in A},axis=1).sort_index()
r=c.pct_change(fill_method=None); market=r.median(axis=1)
beta=r.rolling(60,min_periods=45).cov(market).div(market.rolling(60,min_periods=45).var(),axis=0)
res=r.sub(beta.mul(market,axis=0))
# A common, observation-free state: high cross-asset dispersion relative to its lagged history.
disp=r.std(axis=1); state=disp.shift(1).gt(disp.rolling(60,min_periods=45).median().shift(1)).astype(float)
# Lagged residual loss, followed by current residual gain; continuous values and 50% state coverage.
loss=(-res.shift(1)).clip(lower=0)
repair=loss*res.clip(lower=0)
rawf=repair.mul(state,axis=0).rolling(10,min_periods=7).mean()
trend=c.div(c.shift(20)).sub(1).div(r.rolling(20,min_periods=15).std().clip(lower=1e-8))
f=pd.DataFrame(np.nan,index=c.index,columns=A)
for dt in c.index:
 q=pd.concat([rawf.loc[dt].rename('x'),trend.loc[dt].rename('trend')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8 and q.x.nunique()>1 and q.trend.nunique()>1:
  b=np.polyfit(q.trend,q.x,1); f.loc[dt,q.index]=q.x-(b[0]*q.trend+b[1])
def ev(h):
 fw=c.shift(-h).div(c).sub(1); vals=[];breadth=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.x.nunique()>1 and q.y.nunique()>1: vals.append((dt,q.x.corr(q.y,method='spearman')));breadth.append(len(q))
 return pd.Series(dict(vals)),breadth
print('FACTOR dispersion_state_conditioned_residual_downside_absorption_10_60obs')
print('EXPRESSION cs_residual(mean_10(I[dispersion(t-1)>median_60(dispersion,t-1)]*max(-residual_return(t-1),0)*max(residual_return(t),0)),risk_adjusted_trend_20)')
print('endpoint',c.index.max().date(),'assets',len(A),'state_days',int(state.sum()))
for h in (1,5,10,20):
 x,n=ev(h); ir=x.mean()/x.std(ddof=1)
 print('HORIZON',h,'IC',round(x.mean(),6),'ICIR',round(ir,6),'hit',round((x>0).mean(),6),'dates',len(x),'mean_n',round(np.mean(n),3),'min_n',min(n),'PASS',abs(x.mean())>=.007 and abs(ir)>=.084)
 for name,mask in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
  z=x[mask];print('REGIME',h,name,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),6))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('QUALITY coverage',round(f.notna().mean().mean(),6),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'turnover',round(np.mean(turns),6),'turnover_comparisons',len(turns),'median_iqr',round(iqr.median(),8))
f.to_pickle('scripts/miner_1_20341109_dispersion_state_conditioned_residual_downside_absorption_10_60obs_signal.pkl')
