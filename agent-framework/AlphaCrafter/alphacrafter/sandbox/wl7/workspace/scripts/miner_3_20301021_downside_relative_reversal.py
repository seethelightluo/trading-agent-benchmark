import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100: F[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
p=pd.concat(F,axis=1).sort_index().ffill(); r=np.log(p).diff()
mom=np.log(p/p.shift(20)); rel=mom.sub(mom.median(axis=1),axis=0)
down=r.where(r<0,0).rolling(20).apply(lambda x: np.sqrt(np.mean(x*x)),raw=True)*np.sqrt(20)
# Contrarian one-day signal: fade downside-adjusted relative 20d trend.
f=-(rel/down.replace(0,np.nan)).shift(1).clip(-8,8)
fr=np.log(p.shift(-1)/p); qs=[]; ns=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
q=pd.Series(qs).dropna(); ic=q.mean(); ir=ic/q.std(ddof=1)*np.sqrt(252)
print(f'dates={len(q)} avg_n={np.mean(ns):.2f} IC={ic:.8f} ICIR={ir:.8f} hit={(q>0).mean():.4f}')
for h in [5,10,20]:
 zq=[]; frh=np.log(p.shift(-h)/p)
 for dt in p.index:
  z=pd.concat([f.loc[dt],frh.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:zq.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('H',h,'IC',np.nanmean(zq),'dates',len(zq))
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(p),'instruments',len(F),'last',p.index.max().date())
for name,sub in [('early',q.iloc[:len(q)//3]),('middle',q.iloc[len(q)//3:2*len(q)//3]),('late',q.iloc[2*len(q)//3:])]: print(name,len(sub),sub.mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20301021_downside_relative_reversal_signal.csv',index=False)
q.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20301021_downside_relative_reversal_ic.csv',index=False)
