import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_account_dict
u=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
u=[x if isinstance(x,str) else x.get('symbol') for x in u]; P={}
for s in u:
 d=get_stock_daily_data(s,5000)
 if d is not None: P[s]=d.set_index('date')['close'].astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change(); v=r.rolling(20).std(); med=v.median(axis=1); base=med.rolling(60).median().shift(1)
# shock reversal: lagged 5d return, inverse risk-scaled, only elevated aggregate volatility
shock=(med.shift(1)>base*1.15); f=-(px.pct_change(5).shift(1)/v.shift(1)).where(shock, np.nan)
fwd=px.shift(-10).div(px)-1; vals=[]; active=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z))); active.append(dt)
r=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date').dropna(); print('universe',len(u),'available',len(P),'active_dates',len(r),'avg_n',r.n.mean()); print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean(),'coverage active',r.n.mean()/len(P))
for h in [1,3,5,10]:
 fw=px.shift(-h).div(px)-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('h',h,'n',len(a),'ic',np.nanmean(a),'ir',np.nanmean(a)/np.nanstd(a,ddof=1))
f.to_csv('scripts/miner_1_20330916_stress_reversal_signal.csv')
