import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'; cut=pd.Timestamp('2027-03-10')
p=pd.DataFrame({s:pd.read_csv(f'{base}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}).sort_index().loc[:cut]
r=p.pct_change(); down=r.where(r<0).rolling(15,min_periods=8).std(); allv=r.rolling(60,min_periods=40).std(); f=(down/allv).shift(1)
def calc(h):
 y=p.shift(-h)/p-1; out=[]
 for d in p.index:
  z=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: out.append((d,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(out,columns=['date','ic','n'])
for h in [1,3,5,10]:
 z=calc(h); print('h',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(z.ic.mean(),5),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4))
z=calc(1); q=f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna(); print('turnover',round(q.mean(),5),'period',z.date.min().date(),z.date.max().date())
for name,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027','2027-01-01','2027-03-10')]:
 q=z[(z.date>=a)&(z.date<=b)]; print('regime',name,'dates',len(q),'IC',round(q.ic.mean(),5) if len(q) else None,'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),5) if len(q)>1 else None)
