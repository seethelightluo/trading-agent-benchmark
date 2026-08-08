import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
# One idea: 60-session range-position persistence. High scores identify assets closing near
# their own 60d high, a scale-free breakout/relative-strength signal across asset classes.
assets=get_account_dict()['watch_list']; frames={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date'])
 frames[a]=d.drop_duplicates('date').set_index('date').sort_index()
idx=pd.DatetimeIndex(sorted(set.intersection(*[set(d.index) for d in frames.values()])))
close=pd.DataFrame({a:frames[a].loc[idx,'close'] for a in assets})
hi=close.rolling(60,min_periods=45).max(); lo=close.rolling(60,min_periods=45).min()
fac=((close-lo)/(hi-lo)).replace([np.inf,-np.inf],np.nan)
print('IDEA range_position_persistence_60obs')
print('VISIBLE',idx.min().date(),idx.max().date(),'assets',len(assets),'factor_cells',int(fac.notna().sum().sum()))
all_ic={}
def calc(h):
 fwd=close.shift(-h)/close-1; out=[]; ns=[]
 for dt in idx:
  z=pd.concat([fac.loc[dt].rename('f'),fwd.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:
   out.append((dt,z.f.corr(z.r,method='spearman'))); ns.append(len(z))
 s=pd.Series(dict(out)); all_ic[h]=s
 ir=s.mean()/s.std(ddof=1) if len(s)>1 and s.std(ddof=1)>0 else np.nan
 print('H',h,'IC',f'{s.mean():.6f}','ICIR',f'{ir:.6f}','hit',f'{(s>0).mean():.4f}','dates',len(s),'mean_n',f'{np.mean(ns):.2f}')
 for name,mask in [('2020_2022',s.index<'2023-01-01'),('2023_2026',(s.index>='2023-01-01')&(s.index<'2027-01-01')),('2027_2030',(s.index>='2027-01-01')&(s.index<'2031-01-01')),('2031_plus',s.index>='2031-01-01'),('latest12m',s.index>=s.index.max()-pd.Timedelta(days=365)),('latest6m',s.index>=s.index.max()-pd.Timedelta(days=183))]:
  q=s[mask]; qi=q.mean()/q.std(ddof=1) if len(q)>1 and q.std(ddof=1)>0 else np.nan
  print(' ',name,'n',len(q),'IC',f'{q.mean():.6f}','ICIR',f'{qi:.6f}','hit',f'{(q>0).mean():.4f}')
for h in [1,5,10,20]: calc(h)
ranks=fac.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 z=ranks.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=fac.quantile(.75,axis=1)-fac.quantile(.25,axis=1)
print('DIAG coverage',f'{fac.notna().mean().mean():.6f}','rank_turnover',f'{np.mean(turns):.6f}','median_iqr',f'{iqr.median():.6f}','constant_cross_sections',int((iqr<=1e-12).sum()))
print('NOVELTY not computed unless same-horizon IC and ICIR gates pass; admission requires maximum library Spearman correlation <0.5000.')
