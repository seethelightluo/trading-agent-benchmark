import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,5000)
   if x is not None and len(x)>=100:return x.assign(date=pd.to_datetime(x.date)).sort_values('date').drop_duplicates('date').set_index('date')
  except Exception: pass
D={s:get(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index(); R=C.pct_change()
v=pd.read_csv('../persistent/index_data/VIX.csv');v.date=pd.to_datetime(v.date); v=v.set_index('date').iloc[:,0].astype(float).reindex(C.index).ffill()
bread=(R>0).mean(axis=1).rolling(5).mean(); r5=C.pct_change(5); res=r5.sub(r5.median(axis=1),axis=0); vol=R.rolling(30).std()*np.sqrt(5)
# Robustness variant: less selective stress activation, lagged and fully observable.
stress=(v>v.rolling(252).quantile(.60))&(bread<.50)
f=(-res/vol.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).where(stress.shift(1)); f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1: rows.append((d,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
# signal turnover as fraction changing rank ordering on consecutive active dates
rank=f.rank(axis=1,pct=True); active=rank.notna().sum(axis=1)>=8
turn=(rank[active].diff().abs().mean(axis=1)>0).mean() if active.any() else np.nan
print('assets',len(D),'ic_dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15,'active_days',int(stress.sum()),'turnover_proxy',turn)
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic; print(a+'-'+b,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std())
q=o.tail(120).ic;print('recent120',q.mean(),q.mean()/q.std())
for h in [1,3,5]:
 fr=C.pct_change(h).shift(-h); z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:z.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
 print('horizon',h,'IC',np.nanmean(z),'dates',len(z))
f.to_csv('scripts/miner_1_20320819_broad_stress_residual_signal.csv',index_label='date')
