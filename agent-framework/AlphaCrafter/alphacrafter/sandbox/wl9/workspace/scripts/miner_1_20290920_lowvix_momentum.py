import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 try:d=get_stock_daily_data(s,days=4000)
 except:d=None
 if d is None or len(d)<250:
  try:d=get_index_daily_data(s,days=4000)
  except:d=None
 return None if d is None else d.set_index(pd.to_datetime(d.date)).close.astype(float).sort_index()
pdct={s:g(s) for s in U};pdct={s:x for s,x in pdct.items() if x is not None};v=g('VIX');p=pd.DataFrame(pdct).sort_index().ffill();r=p.pct_change(); gate=v.reindex(p.index).ffill()<v.reindex(p.index).ffill().rolling(60,min_periods=60).median();ret=p.pct_change(20);vol=r.rolling(20).std();f=(ret/(vol*np.sqrt(20))).shift(1);fr=p.pct_change(10).shift(-10);a=[]
for d in f.index:
 if not bool(gate.get(d,False)):continue
 z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(z)>=8:a.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
a=pd.DataFrame(a,columns=['date','ic','n']).set_index('date');print('dates',len(a),'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(),'hit',(a.ic>0).mean(),'coverage',a.n.mean()/15,'turnover',f.rank(pct=True).loc[a.index].diff().abs().mean(axis=1).mean());print('recent252',a.tail(252).ic.mean(),a.tail(252).ic.mean()/a.tail(252).ic.std())
