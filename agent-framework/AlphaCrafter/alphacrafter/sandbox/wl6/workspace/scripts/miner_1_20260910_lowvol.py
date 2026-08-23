import numpy as np,pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-09-09'); b='../persistent/stock_data'
P=pd.DataFrame({s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date')['close'] for s in U}).sort_index().loc[:cut]; R=P.pct_change(); f=-R.rolling(20,min_periods=15).std();
for h in [1,5,10]:
 y=P.shift(-h)/P-1; vals=[]
 for dt in P.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=pd.DataFrame(vals,columns=['date','ic','n']).set_index('date'); print('horizon',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),6),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),6),'hit',round((a.ic>0).mean(),4))
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   q=a[(a.index.year>=lo)&(a.index.year<=hi)].ic; print('regime',lo,hi,'dates',len(q),'ICIR',round(q.mean()/q.std(ddof=1),6))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),6))
