import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={s:get_stock_daily_data(s,days=5000) for s in U}
px={s:x.set_index('date')['close'].astype(float) for s,x in raw.items() if x is not None and len(x)>100}
P=pd.DataFrame(px).sort_index(); ret=P.shift(1)/P.shift(21)-1; r=P.pct_change().shift(1)
down=r.where(r<0,0).rolling(40,min_periods=25).std(); F=ret/down.replace(0,np.nan)
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; vals=[]; ns=[]
 for d in F.index:
  a=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: vals.append(a.iloc[:,0].corr(a.iloc[:,1])); ns.append(len(a))
 ser=pd.Series(vals); print('H',h,'dates',len(ser),'avgN',np.mean(ns),'IC',ser.mean(),'ICIR',ser.mean()/ser.std(ddof=1),'hit',(ser>0).mean())
 if h==20: print('thirds',*[ser.iloc[i*len(ser)//3:(i+1)*len(ser)//3].mean() for i in range(3)])
print('coverage',F.notna().sum().sum()/(F.shape[0]*len(U))); print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
z=F.stack().rename('signal').reset_index(); z.columns=['date','symbol','signal']; z.to_csv('scripts/miner_1_20330207_downside_momentum_signal.csv',index=False); print('dates',F.index.min(),F.index.max(),'assets',len(px),'artifact rows',len(z))
