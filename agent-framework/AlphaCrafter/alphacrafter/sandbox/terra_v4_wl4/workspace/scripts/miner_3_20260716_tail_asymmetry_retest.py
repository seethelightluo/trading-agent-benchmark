import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}; P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date')['close'].loc[:'2026-07-15'].sort_index()
 rr=d.pct_change(); up=rr.clip(lower=0).rolling(40,min_periods=25).mean(); dn=(-rr.clip(upper=0)).rolling(40,min_periods=25).mean()
 F[s]=(up-dn)/(up+dn+1e-12); P[s]=d
f=pd.concat(F,axis=1).sort_index(); p=pd.concat(P,axis=1).sort_index()
print('rows',len(p),'assets',len(U),'date',p.index.min(),p.index.max())
for h in [1,5,10]:
 vals=[];ns=[];dates=[]
 for dt in f.index:
  xs=[];ys=[]
  for s in U:
   if dt in F[s].index:
    j=P[s].index.searchsorted(dt)
    if j<len(P[s].index) and P[s].index[j]==dt and j+h<len(P[s]):
     xs.append(F[s].loc[dt]);ys.append(P[s].iloc[j+h]/P[s].iloc[j]-1)
  z=pd.DataFrame({'x':xs,'y':ys}).dropna()
  if len(z)>=8: vals.append(spearmanr(z.x,z.y).statistic);ns.append(len(z));dates.append(dt)
 a=np.array(vals); ser=pd.Series(a,index=dates)
 print(h,'N',len(a),'avgN',round(np.mean(ns),2),'coverage',round(sum(ns)/(len(a)*15),4),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'year',ser.groupby(ser.index.year).mean().round(4).to_dict())
print('turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean()*2,4))
for name,x in [('rev',pd.concat({s:-P[s].pct_change(5) for s in U},axis=1)),('mom',pd.concat({s:P[s].pct_change(20)/P[s].pct_change().rolling(20).std() for s in U},axis=1))]: print('corr',name,round(f.stack().corr(x.stack()),4))
