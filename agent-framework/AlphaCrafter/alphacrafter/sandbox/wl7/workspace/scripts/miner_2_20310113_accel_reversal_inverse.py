import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:F[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
p=pd.concat(F,axis=1).sort_index().ffill();r=np.log(p).diff();v=r.rolling(40,min_periods=20).std()*np.sqrt(252)
f=((r.rolling(20,min_periods=20).sum()-r.rolling(5,min_periods=5).sum())/v.replace(0,np.nan)).shift(1)
f=f.sub(f.median(axis=1),axis=0).clip(-8,8)
for h in [1,3,5,10,20]:
 fr=np.log(p.shift(-h)/p);q=[];n=[];ds=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));n.append(len(z));ds.append(dt)
 q=pd.Series(q,index=ds).dropna();print(f'H{h} dates={len(q)} avg_n={np.mean(n):.2f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.8f} hit={(q>0).mean():.4f}')
 if h==10:q.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_2_20310113_accel_reversal_inverse_ic.csv',index=False)
fr=np.log(p.shift(-10)/p);q=[];ds=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:q.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(dt)
q=pd.Series(q,index=ds).dropna();m=len(q);print('regimes',*[f'{q.iloc[a:b].mean():.8f}' for a,b in [(0,m//3),(m//3,2*m//3),(2*m//3,m)]])
print('recent252',q.tail(252).mean(),'recent756',q.tail(756).mean());print('coverage',f.notna().sum().sum()/(len(f)*15),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(p),'instruments',len(F))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20310113_accel_reversal_inverse_signal.csv',index=False)
