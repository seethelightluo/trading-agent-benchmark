import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; px={}
for a in A:
 f=f'{b}/{a}.csv'
 if os.path.exists(f): px[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().close
P=pd.DataFrame(px).sort_index(); r=P.pct_change(); dn=r.where(r<0)
dv=dn.rolling(40,min_periods=25).std(); f=(P.pct_change(20)/(dv*np.sqrt(20)+1e-12)).shift(1)
for h in [1,5,10,20]:
 z=[];ns=[];ds=[]
 for d in f.index:
  x=f.loc[d]; y=(P.shift(-h)/P-1).loc[d]; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   q=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(q):z.append(q);ns.append(ok.sum());ds.append(d)
 s=pd.Series(z,index=ds); q=s.tail(250)
 print('horizon',h,'dates',len(s),'avg_n',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4),'recent250',round(q.mean(),6),round(q.mean()/q.std(ddof=1),6))
print('assets',len(px),'rows',len(P),'valid_factor_rate',round(f.notna().sum().sum()/(len(f)*len(px)),4))
