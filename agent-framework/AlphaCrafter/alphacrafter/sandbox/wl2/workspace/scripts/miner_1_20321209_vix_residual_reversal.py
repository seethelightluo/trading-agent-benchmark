import numpy as np,pandas as pd,os
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 for fn in(get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,5000)
   if x is not None and len(x)>100:return x.assign(date=pd.to_datetime(x.date)).sort_values('date').drop_duplicates('date').set_index('date')
  except:pass
D={s:load(s) for s in U};D={s:x for s,x in D.items() if x is not None};C=pd.DataFrame({s:x.close.astype(float) for s,x in D.items()}).sort_index();R=C.pct_change();med=R.median(axis=1);res=R.sub(med,axis=0);vol=R.rolling(20,min_periods=15).std()
try:
 v=pd.read_csv('../persistent/index_data/VIX.csv');v['date']=pd.to_datetime(v['date']);v=v.set_index('date').close.astype(float).reindex(C.index).ffill();stress=(v.pct_change(5)>0)&(v>v.rolling(60,min_periods=30).median())
except Exception as e: print(e);stress=pd.Series(False,index=C.index)
f=-(res.rolling(5,min_periods=4).sum()/vol);f=f.mul(stress.astype(float),axis=0).replace([np.inf,-np.inf],np.nan).shift(1)
rows=[]
for d in f.index:
 q=pd.concat([f.loc[d],R.shift(-1).loc[d]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1:rows.append((d,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
o=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('assets',len(D),'dates',len(o),'avg_n',round(o.n.mean(),3),'coverage',round(o.n.mean()/15,4));print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()));print('stress_dates',int(stress.sum()));
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a+'-'+b,len(q),round(q.mean(),6),round(q.mean()/q.std(),6))
print('turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()),4));f.to_csv('scripts/miner_1_20321209_vix_residual_reversal_signal.csv',index_label='date')
