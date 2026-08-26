import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; end=pd.Timestamp('2028-08-14')
F={}; FR={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=end].set_index('date'); c=d.close
 r=c.pct_change(); f=-(c.shift(1)/c.shift(6)-1)/(r.shift(1).rolling(20,min_periods=15).std()+1e-8)
 F[s]=f; FR[s]={h:c.shift(-h)/c-1 for h in [1,5,10,20]}
f=pd.DataFrame(F); print('date range',f.index.min().date(),f.index.max().date())
for h in [1,5,10,20]:
 z=[]; ns=[]
 for dt in f.index:
  a=pd.DataFrame({'f':f.loc[dt], 'fr':pd.Series({s:FR[s][h].get(dt,np.nan) for s in U})}).dropna()
  if len(a)>=8:z.append(spearmanr(a.f,a.fr).statistic);ns.append(len(a))
 z=np.array(z); recent=z[-252:]
 print('h',h,'IC %.6f ICIR %.4f dates %d avgN %.2f recentIC %.6f recentICIR %.4f'%(z.mean(),z.mean()/z.std(ddof=1),len(z),np.mean(ns),recent.mean(),recent.mean()/recent.std(ddof=1)))
rank=f.rank(axis=1,pct=True);print('turnover',rank.diff().abs().mean(axis=1).dropna().mean(),'coverage',f.notna().sum().sum()/f.size)
# regime by IC date
for a,b in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2028-08-14')]:
 z=[]
 for dt in f.loc[a:b].index:
  x=pd.DataFrame({'f':f.loc[dt],'fr':pd.Series({s:FR[s][10].get(dt,np.nan) for s in U})}).dropna()
  if len(x)>=8:z.append(spearmanr(x.f,x.fr).statistic)
 z=np.array(z);print('regime',a,b,'n',len(z),'IC10',z.mean(),'ICIR',z.mean()/z.std(ddof=1))
