import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
d={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): f='../persistent/index_data/'+s+'.csv'
 x=pd.read_csv(f);x.date=pd.to_datetime(x.date);d[s]=x.set_index('date')[['open','close','high','low']]
# range-position shock: prior session close relative to its high-low range, scaled by 20d range; fade close extremes
P=pd.DataFrame({s:d[s]['close'] for s in U}); H=pd.DataFrame({s:d[s]['high'] for s in U}); L=pd.DataFrame({s:d[s]['low'] for s in U}); O=pd.DataFrame({s:d[s]['open'] for s in U})
rng=(H-L).rolling(20).mean(); loc=(P-L)/(H-L).replace(0,np.nan)-.5
f=(-loc/(rng/P)).shift(1)
fr=P.shift(-1)/P-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
x=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(x),'avg_n',x.n.mean(),'coverage',x.n.sum()/len(x)/15);print('IC',x.ic.mean(),'ICIR',x.ic.mean()/x.ic.std(ddof=1),'hit',(x.ic>0).mean());print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
for a,b in [('2020','2022'),('2023','2024'),('2025','2027')]:
 q=x.loc[a:b].ic;print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
out=f.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('scripts/miner_3_20270423_range_location_signal.csv',index=False)
