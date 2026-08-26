import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d): raw[s]=d[['date','close']].drop_duplicates('date').assign(date=lambda x:pd.to_datetime(x.date)).set_index('date').close
px=pd.DataFrame(raw).sort_index().ffill().loc[:pd.Timestamp('2034-02-19')]
r5=px.pct_change(5); v10=px.pct_change().rolling(10).std(); v60=px.pct_change().rolling(60).std()
f=(-(r5)*(v60/v10).clip(.5,3)).where(v10 < v60*.95).shift(1)
fw=px.shift(-10)/px-1; rows=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(q)>=8: rows.append((dt,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('range',px.index.min().date(),px.index.max().date(),'dates',len(r),'avgN',r.n.mean(),'coverage_dates',len(r)/(len(px)-10))
print('IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean())
parts=np.array_split(np.arange(len(r)),3)
print('thirds',[round(r.iloc[p]['ic'].mean(),6) for p in parts],'recent120',r.tail(120).ic.mean(),r.tail(120).ic.mean()/r.tail(120).ic.std(ddof=1))
rr=[]
for i in range(1,len(f)):
 q=pd.concat([f.iloc[i-1],f.iloc[i]],axis=1).dropna()
 if len(q)>=8: rr.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('turnover',np.nanmean(rr),'asset_coverage',f.notna().sum(axis=1).mean()/len(U))
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_1_20340220_broad_compression_reversal_signal.csv')
