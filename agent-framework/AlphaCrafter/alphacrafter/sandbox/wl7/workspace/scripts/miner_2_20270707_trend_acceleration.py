import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
    d=get_stock_daily_data(s, 3000)
    if d is None or len(d)<150: d=get_index_daily_data(s,3000)
    if d is None: continue
    x=d[['date','close']].copy(); x['symbol']=s; rows.append(x)
px=pd.concat(rows).pivot(index='date',columns='symbol',values='close').sort_index()
# lagged 60d trend acceleration, scaled by 60d volatility: long-term trend minus recent trend
ret=px.pct_change()
r20=px/px.shift(20)-1; r60=px/px.shift(60)-1
vol60=ret.rolling(60).std()*np.sqrt(252)
f=(r60-r20)/vol60
# one-day lag and forward 10 trading-day return
sig=f.shift(1); fw=px.shift(-10)/px-1
ics=[]; vals=[]
for dt in sig.index:
    z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
    if len(z)>=8:
        ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); ics.append((dt,ic,len(z)))
D=pd.DataFrame(ics,columns=['date','ic','n']).set_index('date').dropna()
print('dates',len(D),'avg_n',D.n.mean(),'coverage',D.n.sum()/(len(D)*15))
print('IC10',D.ic.mean(),'ICIR',D.ic.mean()/D.ic.std(),'hit',(D.ic>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=D.loc[a:b]; print(a,b,len(q),q.ic.mean() if len(q) else np.nan)
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; aa=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: aa.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,np.nanmean(aa))
D.to_csv('scripts/miner_2_20270707_trend_acceleration_signal.csv')
