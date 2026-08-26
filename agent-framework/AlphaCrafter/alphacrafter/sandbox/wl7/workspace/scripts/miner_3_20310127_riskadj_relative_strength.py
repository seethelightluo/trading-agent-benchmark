import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:F[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
p=pd.concat(F,axis=1).sort_index().ffill(); r=np.log(p).diff()
ret5=p.pct_change(5); vol20=r.rolling(20,min_periods=15).std(); breadth=(ret5>0).sum(axis=1)/ret5.notna().sum(axis=1)
raw=ret5/vol20.replace(0,np.nan)
gate=1.0-0.5*((breadth<0.25)|(breadth>0.80)).astype(float)
f=raw.mul(gate,axis=0).sub(raw.mul(gate,axis=0).median(axis=1),axis=0).shift(1)
for h in [1,5,10,20]:
 qs=[];ns=[];ds=[]; fr=np.log(p.shift(-h)/p)
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(qs,index=ds).dropna(); print(f'H{h} dates={len(q)} avg_n={np.mean(ns):.2f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.8f} hit={(q>0).mean():.4f}')
 if h==1:q.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20310127_riskadj_relative_strength_ic.csv',index=False)
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(p),'instruments',len(F))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310127_riskadj_relative_strength_signal.csv',index=False)
