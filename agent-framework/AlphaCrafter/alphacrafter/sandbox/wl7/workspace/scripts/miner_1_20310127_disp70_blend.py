import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100:F[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
p=pd.concat(F,axis=1).sort_index().ffill(); r=np.log(p).diff()
vol=r.rolling(20,min_periods=15).std(); z=r.div(vol.replace(0,np.nan))
# Relative shock reversal, enabled only in unusually dispersed cross-sections (70th percentile).
shock=-(.7*z.rolling(3,min_periods=3).mean()+.3*z.rolling(10,min_periods=10).mean())
disp=r.std(axis=1).rolling(80,min_periods=40).mean()
gate=(disp>disp.rolling(252,min_periods=80).quantile(.70)).astype(float)
raw=shock.mul(gate,axis=0)
f=raw.sub(raw.median(axis=1),axis=0).shift(1).clip(-8,8)
for h in [1,5,10]:
 qs=[];ns=[];ds=[]; fr=np.log(p.shift(-h)/p)
 for dt in p.index:
  x=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:qs.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'));ns.append(len(x));ds.append(dt)
 q=pd.Series(qs,index=ds).dropna();print(f'H{h} dates={len(q)} avg_n={np.mean(ns):.2f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(252):.8f} hit={(q>0).mean():.4f}')
 if h==1:q.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_1_20310127_disp70_blend_ic.csv',index=False)
fr=np.log(p.shift(-1)/p);qs=[];ds=[]
for dt in p.index:
 x=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(x)>=8 and x.iloc[:,0].nunique()>1 and x.iloc[:,1].nunique()>1:qs.append(x.iloc[:,0].corr(x.iloc[:,1],method='spearman'));ds.append(dt)
q=pd.Series(qs,index=ds).dropna();n=len(q)
print('regimes',*[f'{q.iloc[a:b].mean():.8f}' for a,b in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
print('recent_252',q.tail(252).mean(),'recent_756',q.tail(756).mean())
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(p),'instruments',len(F))
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_1_20310127_disp70_blend_signal.csv',index=False)
