import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for a in A:
    try:
        x=get_stock_daily_data(a,days=2000)
        if x is not None: D[a]=x.set_index('date').close.astype(float)
    except Exception as e: print('ERR',a,e)
p=pd.concat(D,axis=1).sort_index().ffill(); r=p.pct_change()
# candidate: 20d momentum adjusted by 60d volatility (risk-adjusted trend)
f=(p/p.shift(20)-1)/(r.rolling(60).std()*np.sqrt(20))
# forward 5/10/20-day returns; cross-sectional IC date observations
for name,z in [('riskmom20',f),('mom10',p/p.shift(10)-1),('reversal5',-(p/p.shift(5)-1)),('lowvol',-r.rolling(20).std())]:
 print('\n',name,'assets',len(D),'dates',len(p))
 for h in [1,5,10,20]:
  ic=[]
  for dt in z.index:
   if dt not in p.index: continue
   ix=p.index.get_loc(dt)
   if ix+h>=len(p): continue
   q=pd.concat([z.loc[dt],(p.iloc[ix+h]/p.iloc[ix]-1).rename('fwd')],axis=1).dropna()
   if len(q)>=8: ic.append(q.iloc[:,0].corr(q.fwd))
  s=pd.Series(ic).dropna(); print('h',h,'n',len(s),'IC',round(s.mean(),4),'ICIR',round(s.mean()/s.std(),4) if s.std()>0 else None,'hit',round((s>0).mean(),3))
 # turnover rank change daily
 rank=z.rank(axis=1,pct=True); print('turn',round(rank.diff().abs().mean(axis=1).mean(),4),'coverage',round(z.notna().sum(axis=1).mean()/len(D),3))
