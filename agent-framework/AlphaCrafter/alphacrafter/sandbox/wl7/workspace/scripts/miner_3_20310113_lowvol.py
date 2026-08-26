import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:F[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
p=pd.concat(F,axis=1).sort_index().ffill(); r=np.log(p).diff()
# Volatility-regime carry proxy: prefer assets with stable low realized volatility,
# but penalize recent drawdown; lagged and cross-sectionally centered.
v=r.rolling(20,min_periods=15).std()*np.sqrt(252)
down=r.clip(upper=0).rolling(20,min_periods=15).std()*np.sqrt(252)
dd=(p/p.rolling(60,min_periods=40).max()-1).clip(upper=0)
raw=-v-0.5*down-0.25*dd
f=raw.sub(raw.median(axis=1),axis=0).shift(1)
for h in [1,5,10,20]:
 qs=[];ns=[];ds=[]; fr=np.log(p.shift(-h)/p)
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(dt)
 q=pd.Series(qs,index=ds).dropna(); print(f'H{h} dates={len(q)} avg_n={np.mean(ns):.2f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.8f} hit={(q>0).mean():.4f}')
 if h==1:q.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20310113_lowvol_ic.csv',index=False)
fr=np.log(p.shift(-1)/p);qs=[];ds=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(dt)
q=pd.Series(qs,index=ds).dropna();n=len(q)
print('regimes',*[f'{q.iloc[a:b].mean():.8f}' for a,b in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
print('recent_252',q.tail(252).mean(),'recent_756',q.tail(756).mean())
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(p),'instruments',len(F))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310113_lowvol_signal.csv',index=False)
