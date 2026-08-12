import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 d=None
 try:d=get_index_daily_data(s,5000)
 except:pass
 if d is None or len(d)<100:
  try:d=get_stock_daily_data(s,5000)
  except:pass
 if d is None:return None
 return d.assign(date=pd.to_datetime(d.date)).sort_values('date').drop_duplicates('date').set_index('date')
D={s:get(s) for s in U};D={s:x for s,x in D.items() if x is not None}
C=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index();R=C.pct_change()
# Lagged stress residual reversal: only activate after lagged broad weakness and high VIX.
vix=pd.read_csv('../persistent/index_data/VIX.csv');vix.date=pd.to_datetime(vix.date);vix=vix.set_index('date').iloc[:,0].astype(float).reindex(C.index).ffill()
bread=(R>0).mean(axis=1).rolling(5).mean(); stress=(vix>vix.rolling(252).quantile(.70))&(bread<.45)
r5=C.pct_change(5);res=r5.sub(r5.median(axis=1),axis=0);vol=R.rolling(30).std()*np.sqrt(5)
f=(-res/vol.replace(0,np.nan)).replace([np.inf,-np.inf],np.nan).where(stress.shift(1)); f=f.sub(f.mean(axis=1),axis=0)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1:rows.append((d,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('assets',len(D),'dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15 if len(o) else 0,'stress_days',int(stress.sum()))
print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a+'-'+b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
print('recent120',o.tail(120).ic.mean(),o.tail(120).ic.mean()/o.tail(120).ic.std())
f.to_csv('scripts/miner_1_20320805_stress_lagged_signal.csv',index_label='date')
