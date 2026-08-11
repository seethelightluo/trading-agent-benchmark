import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date') for s in U}
rows=[]
for s,d in D.items():
 d=d.loc[:'2027-07-28']; p=d.shift(1)
 # gap from prior close to current open, known only after prior completed day? lag one day
 gap=p.open/p.close.shift(1)-1
 fac=-gap/(p.close.pct_change().rolling(10).std()+.002)
 r=d.close.shift(-1)/d.close-1
 rows.append(pd.DataFrame({'f':fac,'r':r},index=d.index))
X=pd.concat(rows).sort_index(); z=[]; ns=[]
for _,g in X.dropna().groupby(level=0):
 if len(g)>=8:z.append(spearmanr(g.f,g.r).statistic);ns.append(len(g))
z=np.array(z);print('dates',len(z),'avgN',np.mean(ns),'IC',np.mean(z),'ICIR',np.mean(z)/np.std(z,ddof=1),'coverage',len(X.dropna())/(len(z)*15),'turnover proxy')
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026','2027')]:
 q=X.loc[a:b];v=[]
 for _,g in q.dropna().groupby(level=0):
  if len(g)>=8:v.append(spearmanr(g.f,g.r).statistic)
 v=np.array(v);print(a,b,len(v),np.mean(v),np.mean(v)/np.std(v,ddof=1))
