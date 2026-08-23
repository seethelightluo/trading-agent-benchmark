import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): px[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close
P=pd.DataFrame(px).sort_index(); R=P.pct_change();
# Range-location reversal: recent return divided by realized range, contrarian; avoids raw volatility scale
hi=P.rolling(20,min_periods=15).max(); lo=P.rolling(20,min_periods=15).min()
loc=(P-hi.shift(1))/(hi.shift(1)-lo.shift(1)+1e-12)
# prefer assets that are near lower range (negative loc), with sign reversed as alpha
f=-loc
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); a=q.ic
 print('H',h,'dates',len(q),'avgN',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,5),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),5))
 print('years',q.groupby(q.index.year).ic.mean().round(4).to_dict())
print('rows',len(P),'instruments',len(P.columns),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
