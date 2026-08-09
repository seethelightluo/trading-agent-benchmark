import numpy as np, pandas as pd, glob, os
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
cut=pd.Timestamp('2027-03-10'); frames={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(p):
  d=pd.read_csv(p,parse_dates=['date']).sort_values('date'); frames[s]=d[d.date<=cut].set_index('date')
px=pd.concat({s:frames[s]['close'] for s in frames},axis=1).sort_index(); r=px.pct_change()
f=px.shift(5)/px.shift(65)-1; vol=r.rolling(20).std()*np.sqrt(252); sig=f/vol
rows=[]
for dt in sig.index:
 for h in [1,5,10]:
  y=px.shift(-h).loc[dt]/px.loc[dt]-1; z=pd.DataFrame({'f':sig.loc[dt],'y':y}).dropna()
  if len(z)>=8: rows.append((dt,h,z.f.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [1,5,10]:
 q=a[a.h==h]; m=q.ic.mean(); sd=q.ic.std(ddof=1); print('H',h,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(m,6),'ICIR',round(m/sd,6),'hit',round((q.ic>0).mean(),4))
 if h==1: print(q.assign(reg=q.date.dt.year.map(lambda x:'2020-22' if x<=2022 else ('2023-24' if x<=2024 else '2025-27'))).groupby('reg').ic.mean())
out=sig.stack().rename('signal').reset_index(); out.columns=['date','symbol','signal']; out.to_csv('scripts/miner_2_20270311_medium_trend_quality_signal.csv',index=False)
print('coverage',sig.notna().sum().sum()/(len(sig)*len(sig.columns)),'symbols',len(frames)); print('turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
