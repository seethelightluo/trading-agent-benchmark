import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b='../persistent/stock_data'; cut=pd.Timestamp('2027-01-27')
p=pd.DataFrame({s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}).sort_index().loc[:cut]
r=p.pct_change()
# One-session lagged short/long realized-vol ratio; available at decision after prior close.
f=(r.rolling(5,min_periods=5).std()/r.rolling(40,min_periods=30).std()).shift(1)
def calc(h):
 y=p.shift(-h)/p-1; o=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   o.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(o,columns=['date','ic','n'])
for h in [1,3,5,10]:
 z=calc(h); print('h',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(z.ic.mean(),5),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4))
z=calc(1)
rank=f.rank(axis=1,pct=True); print('turnover',round(rank.diff().abs().mean(axis=1).dropna().mean(),5),'period',z.date.min().date(),z.date.max().date())
for a,bucket in [('2020-2022',('2020-01-01','2022-12-31')),('2023-2024',('2023-01-01','2024-12-31')),('2025-2027',('2025-01-01','2027-01-27'))]:
 q=z[(z.date>=bucket[0])&(z.date<=bucket[1])]; print('regime',a,'dates',len(q),'IC',round(q.ic.mean(),5) if len(q) else None,'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5) if len(q)>1 else None)
