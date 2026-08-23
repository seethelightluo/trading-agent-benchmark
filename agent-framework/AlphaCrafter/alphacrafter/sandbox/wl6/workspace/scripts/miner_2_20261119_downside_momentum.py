import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b='../persistent/stock_data'; cut=pd.Timestamp('2026-11-18')
px={}
for s in U:
 d=pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:cut]
 px[s]=d.close
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# downside-risk adjusted momentum: prior 20-session return divided by prior downside deviation
mom=p.pct_change(20); down=r.where(r<0,0).rolling(20,min_periods=15).std()
f=(mom/down).shift(1)
def calc(h):
 y=p.shift(-h)/p-1; rows=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(rows,columns=['date','ic','n'])
for h in [1,3,5,10]:
 z=calc(h); print('h',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(z.ic.mean(),5),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4))
z=calc(1)
for lo,hi in [('2020','2022'),('2023','2024'),('2025','2026')]:
 q=z[(z.date>=lo)&(z.date<=hi)].ic; print('regime',lo,hi,'dates',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),5),'period',z.date.min().date(),z.date.max().date())
