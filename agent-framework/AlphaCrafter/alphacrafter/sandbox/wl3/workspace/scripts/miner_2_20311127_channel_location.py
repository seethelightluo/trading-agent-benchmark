import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U},axis=1).sort_index().loc[:'2031-11-27']
# 60-day channel location, lagged; test contrarian distance from high/low (range breakout continuation)
lag=D.shift(1); lo=lag.rolling(60,min_periods=40).min(); hi=lag.rolling(60,min_periods=40).max()
fac=(lag-lo)/(hi-lo)
# center and transform: channel location, with mild 20d trend confirmation
fac=(fac-.5)*(1+abs(lag.pct_change(20)))
fwd=D.shift(-1)/D-1
rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('candidate=channel_location_60d dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15)
print('IC %.8f ICIR %.8f hit %.4f turnover %.5f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean(),fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean()))
for name,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-27','2026-01-01','2027-12-31'),('2028-30','2028-01-01','2030-12-31'),('2031','2031-01-01','2031-11-27'),('recent120',None,None)]:
 q=r.tail(120) if name=='recent120' else r.loc[a:b]
 print(name,len(q),'IC %.8f ICIR %.8f hit %.4f'%(q.ic.mean(),q.ic.mean()/q.ic.std() if len(q)>1 else np.nan,(q.ic>0).mean() if len(q) else np.nan))
fac.loc[r.index].to_csv('scripts/miner_2_20311127_channel_location_signal.csv')
print('signal scripts/miner_2_20311127_channel_location_signal.csv')
