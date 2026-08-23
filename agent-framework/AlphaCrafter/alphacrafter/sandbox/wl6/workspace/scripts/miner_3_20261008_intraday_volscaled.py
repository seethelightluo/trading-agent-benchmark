import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; B='../persistent/stock_data'; S={}
for a in A:
 p=f'{B}/{a}.csv'
 if not os.path.exists(p):continue
 d=pd.read_csv(p);d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date'); ret=d.close/d.close.shift(1)-1
 intr=d.close/d.open-1; vol=ret.rolling(20,min_periods=15).std().shift(1)
 S[a]=pd.DataFrame({'f':-intr/vol,'c':d.close})
for h in [1,5,10]:
 rows=[]
 for a,x in S.items():
  q=x.assign(r=x.c.shift(-h)/x.c-1).dropna();rows += [(t,a,f,r) for t,f,r in zip(q.index,q.f,q.r)]
 d=pd.DataFrame(rows,columns=['date','a','f','r']);v=[];ns=[]
 for t,g in d.groupby('date'):
  g=g.replace([np.inf,-np.inf],np.nan).dropna()
  if len(g)>=8 and g.f.nunique()>1 and g.r.nunique()>1:v.append(spearmanr(g.f,g.r).statistic);ns.append(len(g))
 v=np.array(v);print(h,'dates',len(v),'avg_n',round(np.mean(ns),2),'IC',round(v.mean(),5),'ICIR',round(v.mean()/v.std(ddof=1),5),'hit',round(np.mean(v>0),4))
