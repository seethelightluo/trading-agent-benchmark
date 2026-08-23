import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; macro='../persistent/index_data'; cutoff=pd.Timestamp('2026-09-23')
dxy=pd.read_csv(f'{macro}/DXY.csv',parse_dates=['date']).set_index('date').sort_index().close.pct_change(); f={}; fw={}
for s in U:
 x=pd.read_csv(f'{b}/{s}.csv',parse_dates=['date']).set_index('date').sort_index(); ret=x.close.pct_change(); z=pd.concat([ret,dxy],axis=1,join='inner').dropna(); z.columns=['a','d']; beta=z.a.rolling(60,min_periods=45).cov(z.d)/z.d.rolling(60,min_periods=45).var().replace(0,np.nan); f[s]=(z.a.rolling(20,min_periods=15).sum()-beta*z.d.rolling(20,min_periods=15).sum()).loc[:cutoff]; fw[s]={h:ret.shift(-1).rolling(h).sum().shift(-(h-1)).loc[:cutoff] for h in [1,5,10,20]}
f=pd.DataFrame(f)
for h in [1,5,10,20]:
 y=pd.DataFrame({s:fw[s][h] for s in U}); obs=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: obs.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)))
 a=pd.DataFrame(obs,columns=['date','ic','n']).set_index('date'); print('H',h,'dates',len(a),'avg_n',round(a.n.mean(),2),'coverage',round(a.n.mean()/15,4),'IC',round(a.ic.mean(),5),'ICIR',round(a.ic.mean()/a.ic.std(ddof=1),5),'hit',round((a.ic>0).mean(),4))
 if h==1:
  for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
   q=a[(a.index.year>=lo)&(a.index.year<=hi)].ic; print('REG',lo,hi,'dates',len(q),'IC',round(q.mean(),5),'ICIR',round(q.mean()/q.std(ddof=1),5))
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean(),5))