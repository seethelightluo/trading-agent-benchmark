import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>80: F[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
p=pd.concat(F,axis=1).sort_index().ffill(); r=np.log(p).diff()
base=-np.log(p/p.shift(3))/(r.rolling(20).std()*np.sqrt(252)).replace(0,np.nan)
disp=r.std(axis=1)
stress=disp>disp.rolling(60,min_periods=30).median()
f=base.where(stress,0.0).shift(1)
fr={h:np.log(p.shift(-h)/p) for h in [1,5,10,20]}
allq={}
for h in [1,5,10,20]:
 qs=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr[h].loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(qs).dropna(); allq[h]=q
 print(f'H{h} dates={len(q)} avg_n={np.mean(ns):.2f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.8f} hit={(q>0).mean():.4f}')
q=allq[1]
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(p),'instruments',len(F),'last',p.index.max().date())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20301118_stress_reversal_signal.csv',index=False)
q.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20301118_stress_reversal_ic.csv',index=False)
print('stress_fraction',stress.mean())
print('regime_ic', [q.iloc[i:j].mean() for i,j in [(0,len(q)//3),(len(q)//3,2*len(q)//3),(2*len(q)//3,len(q))]])
