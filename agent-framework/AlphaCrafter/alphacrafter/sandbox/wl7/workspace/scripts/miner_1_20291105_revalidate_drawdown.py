import numpy as np,pandas as pd
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2029-11-03'); b=Path('../persistent/stock_data')
P=pd.concat([pd.read_csv(b/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].rename(s) for s in U],axis=1).sort_index().loc[:end]
dd20=P/P.rolling(20,min_periods=15).max()-1; dd60=P/P.rolling(60,min_periods=40).max()-1; v=np.log(P).diff().rolling(20,min_periods=15).std(); s=((dd20-dd60)/v).shift(1)
y=np.log(P.shift(-10)/P)
rows=[]
for d in s.index:
 z=pd.concat([s.loc[d],y.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1]),len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,x in [('full',r),('recent',r.loc['2028-10-01':]),('last90',r.tail(90))]: print(label,len(x),x.n.mean(),x.ic.mean(),x.ic.mean()/x.ic.std(ddof=1),(x.ic>0).mean())
print('coverage',s.notna().sum(axis=1).mean()/15)
