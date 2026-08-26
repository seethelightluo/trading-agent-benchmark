import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:
  x=d[['date','close']].drop_duplicates('date').set_index('date').close; F[s]=x
p=pd.concat(F,axis=1).sort_index().ffill(); r=np.log(p).diff()
# Relative momentum: 20d return relative to contemporaneous cross-asset median, normalized by 20d vol; lag one day
mom=np.log(p/p.shift(20)); rel=mom.sub(mom.median(axis=1),axis=0); vol=r.rolling(20).std()*np.sqrt(20)
f=(rel/vol).shift(1).clip(-5,5)
out=[]
for h in [1,5,10,20]:
 fr=np.log(p.shift(-h)/p); qs=[]; ns=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1: qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(qs).dropna(); print(f'H={h} dates={len(q)} avg_n={np.mean(ns):.2f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.8f} hit={(q>0).mean():.4f}')
 if h==1:
  q1=q
cov=f.notna().sum().sum()/(len(f)*len(U)); turn=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()
print(f'coverage={cov:.6f} rank_turnover={turn:.6f} rows={len(p)} instruments={len(F)} last={p.index.max().date()}')
n=len(q1)
for name,sub in [('early',q1.iloc[:n//3]),('middle',q1.iloc[n//3:2*n//3]),('late',q1.iloc[2*n//3:])]: print(name,len(sub),sub.mean())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20301007_relative_momentum20_signal.csv',index=False)
q1.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20301007_relative_momentum20_ic.csv',index=False)
