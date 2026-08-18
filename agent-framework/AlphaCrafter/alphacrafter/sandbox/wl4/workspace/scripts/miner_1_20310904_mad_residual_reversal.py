import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
px={s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None and len(d)>300}
P=pd.DataFrame(px).sort_index(); R=P.pct_change()
r10=P.pct_change(10); med=r10.median(axis=1); resid=r10.sub(med,axis=0)
mad=R.rolling(40).apply(lambda x: np.median(np.abs(x-np.median(x))),raw=True)
f=(-resid/(mad+1e-8)).shift(1)
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; vals=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); ic=a.ic.mean(); sd=a.ic.std(ddof=1)
 print('H',h,'dates',len(a),'avgN',a.n.mean(),'IC',ic,'ICIR',ic/sd*np.sqrt(252),'hit',(a.ic>0).mean(),'recent',[(n,a.tail(n).ic.mean(),a.tail(n).ic.mean()/a.tail(n).ic.std(ddof=1)*np.sqrt(252)) for n in [365,730,1095]])
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',f.rank(pct=True).diff().abs().mean().mean(),'end',P.index[-1])
