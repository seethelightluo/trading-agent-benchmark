import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
b='../persistent/stock_data'; cut=pd.Timestamp('2026-12-02')
p=pd.DataFrame({s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date').sort_index().close for s in U}).sort_index().loc[:cut]
r=p.pct_change(); f=r.rolling(10,min_periods=8).std().shift(1)
def calc(h):
 y=p.shift(-h)/p-1; o=[]
 for dt in p.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:o.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 return pd.DataFrame(o,columns=['date','ic','n'])
for h in [1,3,5,10]:
 z=calc(h); print('h',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'coverage',round(z.n.mean()/15,4),'IC',round(z.ic.mean(),5),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),5),'hit',round((z.ic>0).mean(),4))
z=calc(1); print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),5),'period',z.date.min().date(),z.date.max().date())
