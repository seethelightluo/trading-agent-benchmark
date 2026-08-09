import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
# One idea: unconditional intraday range-compression persistence, a broad counterpart to conditional downside compression.
assets=get_account_dict()['watch_list']; frames={}
for a in assets:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d['date']); d=d.drop_duplicates('date').set_index('date').sort_index()
 frames[a]=d
idx=sorted(set.intersection(*[set(x.index) for x in frames.values()])); idx=pd.DatetimeIndex(idx)
# range normalized by prior close, then 20/60 mean ratio; negative means compressed recent range.
rng=pd.DataFrame({a:(frames[a].loc[idx,'high']-frames[a].loc[idx,'low']).div(frames[a].loc[idx,'close'].shift(1)).replace([np.inf,-np.inf],np.nan) for a in assets})
fac=-np.log(rng.rolling(20,min_periods=15).mean()/rng.rolling(60,min_periods=45).mean())
close=pd.DataFrame({a:frames[a].loc[idx,'close'] for a in assets})
print('VISIBLE',idx.min().date(),idx.max().date(),'assets',len(assets),'factor cells',int(fac.notna().sum().sum()))
def metrics(h):
 fwd=close.shift(-h)/close-1; ics=[]; dates=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt].rename('f'),fwd.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1: ics.append(z.f.corr(z.r,method='spearman'));dates.append(dt)
 s=pd.Series(ics,index=dates); ir=s.mean()/s.std(ddof=1) if s.std(ddof=1)>0 else np.nan
 print('H',h,'IC',round(s.mean(),6),'ICIR',round(ir,6),'hit',round((s>0).mean(),4),'dates',len(s),'mean_n',round(np.mean([len(pd.concat([fac.loc[d].rename('f'),fwd.loc[d].rename('r')],axis=1).dropna()) for d in dates]),2))
 for label,mask in [('2026_2029',(s.index>='2026-01-01')&(s.index<'2030-01-01')),('2030_plus',s.index>='2030-01-01'),('latest12m',s.index>=s.index.max()-pd.Timedelta(days=365)),('latest6m',s.index>=s.index.max()-pd.Timedelta(days=183))]:
  q=s[mask]; print(' ',label,'n',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6) if len(q)>1 else None,'hit',round((q>0).mean(),4))
for h in [1,5,10,20]:metrics(h)
ranks=fac.rank(axis=1,pct=True); turns=[]
for i in range(1,len(ranks)):
 z=ranks.iloc[[i-1,i]].T.dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
iqr=fac.quantile(.75,axis=1)-fac.quantile(.25,axis=1)
print('DIAG coverage',round(float(fac.notna().mean().mean()),6),'turnover',round(float(np.mean(turns)),6),'iqr',round(float(iqr.median()),6),'constant',int((iqr<=1e-12).sum()))
print('NOVELTY NOT COMPUTED: candidate must be compared with reconstructed signals for all 29 active library factors before admission; absent evidence fails admission.')
