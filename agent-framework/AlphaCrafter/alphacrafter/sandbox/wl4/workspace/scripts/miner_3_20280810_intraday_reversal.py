import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for a in A:
 f='../persistent/stock_data/'+a+'.csv'
 if os.path.exists(f):
  x=pd.read_csv(f,parse_dates=['date']).set_index('date'); D[a]=x
cut=pd.Timestamp('2028-08-10'); D={a:x.loc[x.index<=cut] for a,x in D.items()}
c=pd.DataFrame({a:x.close for a,x in D.items()});o=pd.DataFrame({a:x.open for a,x in D.items()});h=pd.DataFrame({a:x.high for a,x in D.items()});l=pd.DataFrame({a:x.low for a,x in D.items()})
# Lagged 3-day average intraday candle efficiency: contrarian to directional candle body / range.
fac=-(((c-o)/(h-l).replace(0,np.nan)).rolling(3).mean().shift(1))
for n in [1,5,10,20]:
 fwd=c.shift(-n)/c-1;s=[];ns=[]
 for d in fac.index:
  z=pd.concat([fac.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):s.append(q);ns.append(len(z))
 s=pd.Series(s); print(n,'dates',len(s),'avgN',round(np.mean(ns),2),'IC',round(s.mean(),5),'ICIR',round(s.mean()/s.std()*np.sqrt(len(s)),5),'hit',round((s>0).mean(),3),'recent250',round(s.tail(250).mean(),5))
r=fac.rank(axis=1,pct=True);print('coverage',round(np.mean(ns)/15,4),'turn',round(r.diff().abs().mean().mean(),5))
