import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; base='../persistent/stock_data'; cut=pd.Timestamp('2026-10-22')
D={s:pd.read_csv(os.path.join(base,s+'.csv'),parse_dates=['date']).set_index('date').sort_index().loc[:cut] for s in U}
F=pd.DataFrame({s:-(D[s].close/D[s].open-1) for s in U}).sort_index().rolling(3,min_periods=2).mean()
P=pd.DataFrame({s:D[s].close for s in U}).sort_index(); rows=[]
for dt in F.index:
 z=pd.concat([F.loc[dt],P.shift(-1).loc[dt]/P.loc[dt]-1],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1: rows.append((dt,len(z),spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
df=pd.DataFrame(rows,columns=['date','n','ic']).dropna(); q=df.ic
print('cutoff',cut,'dates',len(q),'avgN',df.n.mean(),'coverage',df.n.mean()/15,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean())
for yr,g in df.groupby(df.date.dt.year):print('year',yr,'IC',g.ic.mean(),'n',len(g))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
