"""Fast validation: USDJPY 60-day up/down beta asymmetry; no persistence without a separate library screen."""
import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];E='2028-05-03'
def x(a,m=0):return pd.read_csv(('../persistent/index_data/' if m else '../persistent/stock_data/')+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:E,'close']
c=pd.concat({a:x(a) for a in A},axis=1);r=c.pct_change();z=x('USDJPY',1).pct_change().reindex(r.index)
def b(mask):
 q=z.where(mask);n=mask.rolling(60).sum();sx=q.rolling(60).sum();den=(q*q).rolling(60).sum()-sx*sx/n
 return pd.DataFrame({a:((q*r[a].where(mask)).rolling(60).sum()-sx*r[a].where(mask).rolling(60).sum()/n)/den for a in A})
f=b(z>0)-b(z<0); dates=f.index[f.notna().sum(1)>=8];print('FACTOR usdjpy_up_down_beta_asymmetry_60obs visible',E,'assets',len(A),'cells',f.notna().sum().sum(),'/',f.size)
O={}
for h in [1,5,10,20]:
 s=[];ns=[];y=c.shift(-h)/c-1
 for t in dates:
  q=pd.concat([f.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:s.append((t,spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic));ns.append(len(q))
 s=pd.Series(dict(s));O[h]=s;print('H',h,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(),6),'hit',round((s>0).mean(),4),'n',round(np.mean(ns),2))
h=max(O,key=lambda k:abs(O[k].mean()*O[k].mean()/O[k].std()));s=O[h];print('SELECTED',h)
for a,b,n in [('2020','2021','2020'),('2021','2023','2021-22'),('2023','2025','2023-24'),('2025','2030','2025-current')]:
 q=s[(s.index>=a)&(s.index<b)];print('REGIME',n,len(q),round(q.mean(),6),round(q.mean()/q.std(),6),round((q>0).mean(),4))
rk=f.rank(axis=1);v=[]
for i in range(1,len(rk)):
 q=pd.concat([rk.iloc[i],rk.iloc[i-1]],axis=1).dropna()
 if len(q)>=8:v.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('TURNOVER',round(np.mean(v),6),'coverage',round(f.notna().mean().mean(),4))
