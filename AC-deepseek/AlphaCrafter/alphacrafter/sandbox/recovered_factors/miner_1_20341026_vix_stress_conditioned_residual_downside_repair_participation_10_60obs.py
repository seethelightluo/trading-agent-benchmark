"""Single idea: VIX-stress-conditioned idiosyncratic downside repair participation.
At each completed date t, measures whether an asset repairs an idiosyncratic downside
shock during lagged elevated VIX conditions, then removes ordinary 20d risk-adjusted trend.
All state gates are lagged one bar; no future observations are used."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data, get_index_daily_data
A=get_account_dict()['watch_list']; raw={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
 raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
c=pd.concat({a:raw[a].close.astype(float) for a in A},axis=1).sort_index()
r=c.pct_change(fill_method=None); mkt=r.median(axis=1)
beta=r.rolling(60,min_periods=45).cov(mkt).div(mkt.rolling(60,min_periods=45).var(),axis=0)
res=r.sub(beta.mul(mkt,axis=0)); z=res.sub(res.rolling(60,min_periods=45).mean()).div(res.rolling(60,min_periods=45).std().clip(lower=1e-8))
# Observation-only VIX is explicitly used only as a common state variable.
v=get_index_daily_data('VIX',5000).copy(); v['date']=pd.to_datetime(v['date']); v=v.drop_duplicates('date').set_index('date').sort_index()
vc=pd.to_numeric(v['close'],errors='coerce').reindex(c.index).ffill()
# State known before today's repair: above trailing 60d median VIX.
stress=vc.shift(1).gt(vc.rolling(60,min_periods=45).median().shift(1)).astype(float)
# Yesterday's idiosyncratic downside magnitude times today's residual repair, averaged over 10d.
rawf=((-z.shift(1)).clip(0,3)*res).mul(stress,axis=0).rolling(10,min_periods=7).mean()
trend=c.div(c.shift(20)).sub(1).div(r.rolling(20,min_periods=15).std().clip(lower=1e-8))
f=pd.DataFrame(np.nan,index=c.index,columns=A)
for dt in c.index:
 q=pd.concat([rawf.loc[dt].rename('x'),trend.loc[dt].rename('t')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(q)>=8 and q.x.nunique()>1 and q.t.nunique()>1:
  b=np.polyfit(q.t,q.x,1); f.loc[dt,q.index]=q.x-(b[0]*q.t+b[1])
def evaluate(h):
 fw=c.shift(-h).div(c).sub(1); out=[]; ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.x.nunique()>1 and q.y.nunique()>1: out.append((dt,q.x.corr(q.y,method='spearman')));ns.append(len(q))
 x=pd.Series(dict(out)); return x,ns
print('FACTOR vix_stress_conditioned_residual_downside_repair_participation_10_60obs')
print('EXPRESSION residualize_cs(mean_10(I[VIX[t-1]>median_60(VIX,t-1)]*clip(-zscore_60(residual_return[t-1]),0,3)*residual_return[t]),risk_adjusted_trend_20)')
print('endpoint',c.index.max().date(),'assets',len(A),'vix_nonmissing',int(vc.notna().sum()))
allres={}
for h in (1,5,10,20):
 x,ns=evaluate(h);allres[h]=x
 print('HORIZON',h,'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),6),'dates',len(x),'mean_n',round(np.mean(ns),3),'min_n',min(ns),'PASS',abs(x.mean())>=.007 and abs(x.mean()/x.std(ddof=1))>=.084)
 for nm,mask in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
  y=x[mask]; print('REGIME',h,nm,'dates',len(y),'IC',round(y.mean(),6),'ICIR',round(y.mean()/y.std(ddof=1),6),'hit',round((y>0).mean(),6))
rk=f.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: ts.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('QUALITY coverage',round(f.notna().mean().mean(),6),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'turnover',round(np.mean(ts),6),'turnover_comparisons',len(ts),'median_iqr',round(iqr.median(),6),'stress_days',int(stress.sum()))
f.to_pickle('scripts/miner_1_20341026_vix_stress_conditioned_residual_downside_repair_participation_10_60obs_signal.pkl')
