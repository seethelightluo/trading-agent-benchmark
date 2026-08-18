import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut='2034-10-27'
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float) for s in U}
px=pd.DataFrame(D).sort_index().ffill().loc[:cut]; r=np.log(px).diff()
# Conditional short-term reversal: fade the last 5d move, but only when the asset's 60d trend is positive;
# normalize by 20d volatility and lag one day. This avoids look-ahead while limiting reversal to constructive regimes.
rev=-r.rolling(5,min_periods=5).sum(); trend=r.rolling(60,min_periods=45).sum(); vol=r.rolling(20,min_periods=15).std()
f=(rev/vol).where(trend>0)
f=f.sub(f.median(axis=1),axis=0).shift(1)
fr={h:np.log(px.shift(-h)/px) for h in [5,10,20,40]}
for h,x in fr.items():
 vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],x.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z))
 q=pd.Series(vals); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
rank=f.rank(axis=1,pct=True); print('coverage',round(f.notna().mean().mean(),6),'turnover',round(rank.diff().abs().mean(axis=1).mean(),6))
for a,b in [('2020','2024'),('2025','2029'),('2030','2034')]:
 vals=[]
 for dt in f.loc[a:b].index:
  z=pd.concat([f.loc[dt],fr[10].loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 q=pd.Series(vals); print('REG',a,b,'dates',len(q),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6))
f.to_csv('scripts/miner_2_20341027_constructive_reversal_signal.csv'); print('signal_path scripts/miner_2_20341027_constructive_reversal_signal.csv')
