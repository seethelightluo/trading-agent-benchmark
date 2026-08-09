"""Candidate: correlation-conditioned residual trend, 20/60 observations.
The factor measures an asset's 20-day return after removing its trailing beta to
common median returns, amplified only when its 20-day average pairwise
correlation is below its own 60-day median. Low-correlation regimes make
idiosyncratic trends more likely to be diversifying rather than a common beta bet.
All inputs end at the score date.
"""
import numpy as np
import pandas as pd
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data

assets=get_account_dict()['watch_list']
raw={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 raw[a]=d.drop_duplicates('date').set_index('date').sort_index()
close=pd.concat({a:raw[a].close.astype(float) for a in assets},axis=1).sort_index()
r=close.pct_change(); m=r.median(axis=1)
# Per-asset trailing beta and residual, using completed returns through score date.
beta=r.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0)
res=r.sub(beta.mul(m,axis=0))
res_trend=res.rolling(20,min_periods=15).sum()
# Average off-diagonal correlation, and a prior 60d median comparator.
def mean_pair_corr(x):
 c=x.corr().values; n=c.shape[0]
 return (c.sum()-np.trace(c))/(n*(n-1))
corr_state=r.rolling(20,min_periods=15).apply(lambda x: mean_pair_corr(pd.DataFrame(x)),raw=False)
# apply receives each column separately under DataFrame rolling; calculate explicit windows instead.
corr_state=pd.Series(index=r.index,dtype=float)
for k in range(19,len(r)):
 w=r.iloc[k-19:k+1].dropna(axis=1,how='all')
 if w.shape[1]>=8: corr_state.iloc[k]=mean_pair_corr(w)
base=corr_state.rolling(60,min_periods=45).median().shift(1)
# Smooth binary regime with a continuous signed distance, clipped to prevent dominance.
scale=corr_state.rolling(60,min_periods=45).std().shift(1).clip(lower=1e-4)
lowcorr=((base-corr_state)/scale).clip(-2,2)
f=res_trend.mul(lowcorr,axis=0).replace([np.inf,-np.inf],np.nan)

def evaluate(h):
 fw=close.shift(-h).div(close).sub(1); out=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('factor'),fw.loc[dt].rename('forward')],axis=1).dropna()
  if len(z)>=8 and z.factor.nunique()>1 and z.forward.nunique()>1:
   out.append((dt,z.factor.corr(z.forward,method='spearman'))); ns.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(ddof=1)
 return x,{'dates':len(x),'ic':x.mean(),'icir':x.mean()/sd,'hit_ratio':(x>0).mean(),'mean_instruments':float(np.mean(ns)),'min_instruments':min(ns)}
print('FACTOR correlation_conditioned_residual_trend_20_60obs')
print('expression sum_20(r_i-beta_60,i*median(r)) * clip((median_60(avg_paircorr_20)-avg_paircorr_20)/std_60(avg_paircorr_20),-2,2)')
print('visible_history',close.index.min().date(),close.index.max().date(),'assets',len(assets))
for h in (1,5,10,20):
 x,stats=evaluate(h); print('HORIZON',h,stats)
 if h==10:
  for name,mask in [('2020_2027',x.index<'2028-01-01'),('2028_2030',(x.index>='2028-01-01')&(x.index<'2031-01-01')),('2031_current',x.index>='2031-01-01'),('latest_6m',x.index>=x.index.max()-pd.Timedelta(days=183))]:
   y=x[mask]; print('REGIME',name,'dates',len(y),'ic',y.mean(),'icir',y.mean()/y.std(ddof=1),'hit_ratio',(y>0).mean())
ranks=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(ranks)):
 z=ranks.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=f.quantile(.75,axis=1)-f.quantile(.25,axis=1)
print('coverage',f.notna().mean().mean(),'valid_cells',int(f.notna().sum().sum()),'of',f.size,'daily_rank_turnover',float(np.mean(turn)),'median_iqr',iqr.median(),'constant_dates',int((iqr<=1e-12).sum()))
print('Novelty audit is required before admission if same-horizon gates pass.')
