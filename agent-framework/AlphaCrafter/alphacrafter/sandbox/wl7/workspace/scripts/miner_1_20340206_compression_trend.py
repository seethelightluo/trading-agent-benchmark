import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s,days=6000)
 if d is not None and len(d): D[s]=d.set_index(pd.to_datetime(d.date)).close
px=pd.DataFrame(D).sort_index().ffill().loc[:'2034-02-03']
r5=px.pct_change(5); v10=px.pct_change().rolling(10).std(); v60=px.pct_change().rolling(60).std()
# compressed-path continuation: medium return favored only when short volatility is compressed
f=(r5*(v60/v10).clip(.5,3)).where(v10 < v60*.85).shift(1)
fw=px.shift(-10)/px-1; rows=[]
for dt in f.index:
 q=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
 if len(q)>=8: rows.append((dt,q.iloc[:,0].corr(q.iloc[:,1],method='spearman'),len(q)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('dates',len(r),'avgN',r.n.mean(),'IC',r.ic.mean(),'ICIR',r.ic.mean()/r.ic.std(ddof=1),'hit',(r.ic>0).mean(),'thirds',[r.loc[x,'ic'].mean() for x in np.array_split(r.index,3)],'recent120',r.tail(120).ic.mean(),r.tail(120).ic.mean()/r.tail(120).ic.std(ddof=1))
print('coverage',f.notna().sum(axis=1).mean()/15,'turnover',np.nanmean([1-f.iloc[i-1].corr(f.iloc[i],method='spearman') for i in range(1,len(f))]))
f.to_csv('scripts/miner_1_20340206_compression_trend_signal.csv')
