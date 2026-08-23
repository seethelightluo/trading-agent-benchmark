import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): px[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
# Compression-adjusted breakout: recent return divided by recent realized volatility,
# then multiplied by the inverse ratio of short/long volatility. Positive means
# directional movement emerging from compressed volatility; all inputs lagged at date.
v10=r.rolling(10,min_periods=8).std(); v40=r.rolling(40,min_periods=25).std()
f=r.rolling(10,min_periods=8).sum()/(v10+1e-12) * (v40/(v10+1e-12)).clip(0.25,4)
print('rows',len(P),'instruments',len(P.columns),'span',P.index.min().date(),P.index.max().date())
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: out.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']).set_index('date'); a=q.ic.dropna()
 print('H',h,'dates',len(a),'avgN',round(q.loc[a.index,'n'].mean(),3),'coverage',round(q.loc[a.index,'n'].mean()/15,6),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),6))
 print('years',q.loc[a.index].groupby(q.loc[a.index].index.year).ic.mean().round(4).to_dict())
print('turnover_proxy',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
