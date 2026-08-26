import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100: F[s]=d[['date','close']].drop_duplicates('date').set_index('date').close
p=pd.concat(F,axis=1).sort_index().ffill(); r=np.log(p).diff()
# Defensive trend quality: medium-term return divided by downside deviation,
# with a drawdown penalty; lagged one session.
down=r.clip(upper=0).rolling(30,min_periods=20).std()*np.sqrt(252)
ret20=np.log(p/p.shift(20))
peak=p.rolling(60,min_periods=40).max()
dd=(p/peak-1).clip(upper=0)
raw=ret20/(down.replace(0,np.nan)+1e-8) + 0.35*dd
f=raw.sub(raw.median(axis=1),axis=0).shift(1).clip(-8,8)
frs=[]
for h in [1,5,10,20]:
 qs=[]; ns=[]; dates=[]; fr=np.log(p.shift(-h)/p)
 for dt in p.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 q=pd.Series(qs,index=dates).dropna(); ic=q.mean(); ir=ic/q.std(ddof=1)*np.sqrt(252)
 print(f'H{h} dates={len(q)} avg_n={np.mean(ns):.2f} IC={ic:.8f} ICIR={ir:.8f} hit={(q>0).mean():.4f}')
 if h==1:q.rename('ic').reset_index().rename(columns={'index':'date'}).to_csv('scripts/miner_3_20310113_sortino_trend_ic.csv',index=False)
fr=np.log(p.shift(-1)/p); qs=[]; dates=[]
for dt in p.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: qs.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt)
q=pd.Series(qs,index=dates).dropna(); n=len(q)
print('regimes',*[f'{q.iloc[a:b].mean():.8f}' for a,b in [(0,n//3),(n//3,2*n//3),(2*n//3,n)]])
print('recent_252',q.tail(252).mean(),'recent_756',q.tail(756).mean())
print('coverage',f.notna().sum().sum()/(len(f)*len(U)),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),'rows',len(p),'instruments',len(F),'first',p.index.min().date(),'last',p.index.max().date())
f.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_3_20310113_sortino_trend_signal.csv',index=False)
