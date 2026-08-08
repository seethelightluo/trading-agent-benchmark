import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
# One idea: residual downside rebound strength.  For each asset, measure its
# average next-session idiosyncratic return after its own idiosyncratic down days
# over a 60-session window.  Positive scores identify assets with repeatable
# downside absorption/rebound rather than simply low downside volatility.
assets=get_account_dict()['watch_list']; frames={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
 frames[a]=d.drop_duplicates('date').set_index('date').sort_index()
idx=pd.DatetimeIndex(sorted(set.intersection(*[set(x.index) for x in frames.values()])))
close=pd.DataFrame({a:frames[a].loc[idx,'close'] for a in assets}); r=close.pct_change()
resid=r.sub(r.median(axis=1),axis=0)
# At t, pair residual returns through t with the return on the following session;
# shift ensures no future data in the score. Require 8 historic down observations.
down=resid.lt(0); next_resid=resid.shift(-1)
rebound=next_resid.where(down).rolling(60,min_periods=8).mean().shift(1)
fac=rebound
print('IDEA residual_downside_rebound_strength_60obs')
print('EXPRESSION rolling_mean_60(residual_return[t+1] | residual_return[t]<0), shifted one session')
print('VISIBLE',idx.min().date(),idx.max().date(),'assets',len(assets),'factor_cells',int(fac.notna().sum().sum()))
def report(s,name,mask):
 q=s[mask]; sd=q.std(ddof=1); ir=q.mean()/sd if len(q)>1 and sd else np.nan
 print(' ',name,'n',len(q),'IC',f'{q.mean():.6f}','ICIR',f'{ir:.6f}','hit',f'{(q>0).mean():.4f}')
def calc(h):
 fwd=close.shift(-h)/close-1; out=[]; ns=[]
 for dt in idx:
  z=pd.concat([fac.loc[dt].rename('f'),fwd.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1: out.append((dt,z.f.corr(z.r,method='spearman'))); ns.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1); ir=s.mean()/sd if len(s)>1 and sd else np.nan
 print('H',h,'IC',f'{s.mean():.6f}','ICIR',f'{ir:.6f}','hit',f'{(s>0).mean():.4f}','dates',len(s),'mean_n',f'{np.mean(ns):.2f}')
 report(s,'2020_2022',s.index<'2023-01-01'); report(s,'2023_2026',(s.index>='2023-01-01')&(s.index<'2027-01-01')); report(s,'2027_2030',(s.index>='2027-01-01')&(s.index<'2031-01-01')); report(s,'2031_plus',s.index>='2031-01-01'); report(s,'latest12m',s.index>=s.index.max()-pd.Timedelta(days=365)); report(s,'latest6m',s.index>=s.index.max()-pd.Timedelta(days=183))
for h in [1,5,10,20]: calc(h)
ranks=fac.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 z=ranks.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=fac.quantile(.75,axis=1)-fac.quantile(.25,axis=1)
print('DIAG coverage',f'{fac.notna().mean().mean():.6f}','rank_turnover',f'{np.mean(turns):.6f}','median_iqr',f'{iqr.median():.6f}','constant_cross_sections',int((iqr<=1e-12).sum()))
print('NOVELTY deferred unless numerical admission and regime checks pass.')
