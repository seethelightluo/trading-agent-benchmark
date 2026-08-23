import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; cutoff=pd.Timestamp('2026-09-09')
raw={s:pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
# Compute each asset on its own trading calendar to avoid asynchronous-calendar dilution.
f={}; r={}
for s,x in raw.items():
 tr=x.high-x.low; exp=tr/tr.rolling(20,min_periods=15).median(); clv=-(2*(x.close-x.low)/tr-1)
 f[s]=(clv*exp).replace([np.inf,-np.inf],np.nan).loc[:cutoff]; r[s]=x.close.pct_change().loc[:cutoff]
f=pd.DataFrame(f); r=pd.DataFrame(r); out=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],r.shift(-1).loc[dt]],axis=1).dropna()
 if len(z)>=8: out.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
a=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
print('dates',len(a),'avg_n',a.n.mean(),'coverage',a.n.mean()/15,'IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(ddof=1),'hit',(a.ic>0).mean())
for hzn in [5,10,20]:
 q=[]
 for dt in f.index:
  y=r.shift(-1).rolling(hzn).sum().shift(-(hzn-1)).loc[dt]; z=pd.concat([f.loc[dt],y],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=pd.Series(q); print('h',hzn,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
 q=a[(a.index.year>=lo)&(a.index.year<=hi)].ic; print('regime',lo,hi,'dates',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
