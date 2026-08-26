import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=2400)
 if d is not None: px[s]=d.set_index(pd.to_datetime(d.date)).close
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change(); v=r.rolling(40).std()*np.sqrt(252)
# medium horizon trend, volatility scaled and cross-sectionally demeaned
f=(P/P.shift(60)-1)/v; f=f.sub(f.median(axis=1),axis=0)
fr=P.shift(-10)/P-1
rows=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date').dropna()
def stat(x): return len(x),round(x.n.mean(),2),round(x.ic.mean(),5),round(x.ic.mean()/x.ic.std(ddof=1),5),round((x.ic>0).mean(),4)
print('dates/assets',len(q),len(P.columns),'all',stat(q),'coverage',q.n.sum()/(len(q)*15))
for a,b in [('2020','2022'),('2023','2025'),('2026','2028'),('2028-09-01','2029-03-12')]:
 x=q.loc[a:b]
 if len(x): print(a,b,stat(x))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
q.to_csv('scripts/miner_1_20290312_medium_trend_ic.csv')
