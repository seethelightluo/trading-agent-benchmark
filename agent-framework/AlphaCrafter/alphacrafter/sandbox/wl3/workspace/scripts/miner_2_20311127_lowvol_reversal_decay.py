import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D=pd.concat({s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').close for s in U},axis=1).sort_index().loc[:'2031-11-27']
lag=D.shift(1); rr=lag.pct_change(); r5=lag.pct_change(5); vol=rr.rolling(20,min_periods=15).std(); csvol=vol.median(axis=1)
f=-(r5/(vol*np.sqrt(5))).mul(1/(1+vol.div(csvol,axis=0)))
for h in [5,10]:
 y=D.shift(-h)/D-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('H',h,'dates',len(r),'avg_n',r.n.mean(),'coverage',r.n.mean()/15,'IC %.8f ICIR %.8f hit %.4f'%(r.ic.mean(),r.ic.mean()/r.ic.std(),(r.ic>0).mean()))
 for name,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-25','2023-01-01','2025-12-31'),('2026-27','2026-01-01','2027-12-31'),('2028-30','2028-01-01','2030-12-31'),('2031','2031-01-01','2031-11-27'),('recent120',None,None)]:
  q=r.tail(120) if name=='recent120' else r.loc[a:b]
  print(name,len(q),'%.8f %.8f'%(q.ic.mean(),q.ic.mean()/q.ic.std()))
