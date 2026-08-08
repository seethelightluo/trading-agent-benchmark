"""Candidate: trend-orthogonal delayed residual-downside repair persistence.
Measures whether an idiosyncratic residual downside shock is followed by persistently
favorable close locations over the next several completed sessions. All inputs end t."""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
assets=get_account_dict()['watch_list']; raw={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a:raw[a].close.astype(float) for a in assets},axis=1).sort_index()
high=pd.concat({a:raw[a].high.astype(float) for a in assets},axis=1).reindex(close.index); low=pd.concat({a:raw[a].low.astype(float) for a in assets},axis=1).reindex(close.index)
r=close.pct_change(); market=r.median(axis=1); beta=r.rolling(60,min_periods=45).cov(market).div(market.rolling(60,min_periods=45).var(),axis=0)
resid=r.sub(beta.mul(market,axis=0)); z=(resid-resid.rolling(60,min_periods=45).mean()).div(resid.rolling(60,min_periods=45).std().clip(lower=1e-8))
clv=((2*close-high-low)/(high-low).replace(0,np.nan)).clip(-1,1)
# A shock t-4..t-1 earns its score only when close-location remains favorable through t.
shock=(-z.shift(4)).clip(0,3)
repair=clv.shift(3).rolling(4,min_periods=3).mean()
rawf=shock.mul(repair).rolling(10,min_periods=7).mean()
trend=close.div(close.shift(20)).sub(1).div(r.rolling(20,min_periods=15).std().clip(lower=1e-8))
f=pd.DataFrame(np.nan,index=close.index,columns=assets)
for dt in f.index:
 q=pd.concat([rawf.loc[dt].rename('x'),trend.loc[dt].rename('t')],axis=1).dropna()
 if len(q)>=8 and q.x.nunique()>1 and q.t.nunique()>1:
  b=np.polyfit(q.t,q.x,1); f.loc[dt,q.index]=q.x-(b[0]*q.t+b[1])
def ev(h):
 fw=close.shift(-h).div(close)-1; out=[]; ns=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.x.nunique()>1 and q.y.nunique()>1: out.append((dt,q.x.corr(q.y,method='spearman'))); ns.append(len(q))
 x=pd.Series(dict(out)); return x,dict(dates=len(x),ic=float(x.mean()),icir=float(x.mean()/x.std(ddof=1)),hit_ratio=float((x>0).mean()),mean_instruments=float(np.mean(ns)),min_instruments=int(min(ns)))
print('FACTOR trend_orthogonal_delayed_residual_downside_repair_persistence_10_60obs')
print('EXPRESSION residualize_cs(mean_10(clip(-residual_return_zscore_60[t-4],0,3)*mean_4(CLV[t-3:t])),risk_adjusted_trend_20)')
print('visible_history',close.index.min().date(),close.index.max().date(),'assets',len(assets))
for h in [1,5,10,20]:
 x,s=ev(h); print('HORIZON',h,s)
 if h==10:
  for name,mask in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
   y=x[mask]; print('REGIME',name,'dates',len(y),'ic',float(y.mean()),'icir',float(y.mean()/y.std(ddof=1)),'hit_ratio',float((y>0).mean()))
ranks=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 q=ranks.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
for name,g in [('risk_adjusted_trend',trend),('unresidualized_delayed_repair',rawf)]:
 q=pd.concat([f.stack().rename('f'),g.stack().rename('g')],axis=1).dropna(); print('PROXY_CORR',name,'cells',len(q),'rho',q.f.corr(q.g,method='spearman'))
print('coverage',float(f.notna().mean().mean()),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'daily_rank_turnover',float(np.mean(turns)),'turnover_comparisons',len(turns),'median_iqr',float(iqr.median()))
