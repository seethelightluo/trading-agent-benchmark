import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=2500)
   if x is not None and len(x)>100:return x[['date','open','close']]
  except:pass
xs={s:fetch(s) for s in U};xs={s:x for s,x in xs.items() if x is not None}
P=pd.concat([x.assign(symbol=s) for s,x in xs.items()]); C=P.pivot(index='date',columns='symbol',values='close').sort_index().ffill(); O=P.pivot(index='date',columns='symbol',values='open').reindex(C.index).ffill()
# lagged 5-session open-close directional persistence: average signed intraday returns, contrarian
intr=(C/O-1); f=(-intr.rolling(5,min_periods=4).mean()).shift(1)
print('instruments',len(C.columns),'period',C.index.min(),C.index.max())
for h in [1,5,10]:
 fr=C.shift(-h)/C-1; vals=[];ns=[]
 for d in f.index:
  z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()));ns.append(len(z))
 a=np.array(vals);print('H',h,'dates',len(a),'avgN',np.mean(ns),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0))
print('coverage',f.notna().sum(axis=1).div(len(U)).mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
