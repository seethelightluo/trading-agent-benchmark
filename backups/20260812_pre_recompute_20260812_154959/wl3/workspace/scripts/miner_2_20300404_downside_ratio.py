import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for s in U:
 d=get_stock_daily_data(s,2600)
 if d is None or len(d)<300: d=get_index_daily_data(s,2600)
 if d is not None and len(d):
  x=d[['date','close']].copy(); x.date=pd.to_datetime(x.date).dt.normalize(); x=x.drop_duplicates('date').set_index('date'); frames[s]=x.close.astype(float)
px=pd.DataFrame(frames).sort_index(); ret=np.log(px).diff()
mu=ret.rolling(60,min_periods=40).sum()
neg2=(ret.clip(upper=0)**2).rolling(60,min_periods=15).mean()
dn=np.sqrt(neg2)
sig=mu/(dn*np.sqrt(60)+1e-9)
f=sig.rank(axis=1,pct=True).sub(.5,axis=0)
for h in [1,3,5,10]:
 fr=ret.shift(-h).rolling(h).sum() if h>1 else ret.shift(-1)
 ics=[]; dates=[]; ns=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   q=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(q): ics.append(q); dates.append(dt); ns.append(len(a))
 z=pd.Series(ics,index=pd.to_datetime(dates)); ic=z.mean(); icir=ic/z.std(ddof=1)*np.sqrt(252)
 print('H',h,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(ic,6),'ICIR',round(icir,6),'hit',round((z>0).mean(),4),'coverage',round(np.mean([len(f.loc[d].dropna())/15 for d in dates]),4))
 for lo,hi in [('2020','2022'),('2023','2025'),('2026','2027'),('2028','2030')]:
  q=z[(z.index>=lo)&(z.index<=hi)]
  if len(q): print(' ',lo,hi,len(q),round(q.mean(),6),round(q.mean()/q.std(ddof=1)*np.sqrt(252),4))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6),'last',px.index.max().date(),'assets',len(px.columns))
f.index.name='date'; f.to_csv('scripts/miner_2_20300404_downside_ratio_signal.csv')
