import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.concat([pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close.rename(a) for a in A],axis=1).sort_index().loc[:'2027-03-10'];R=P.pct_change();vol=R.rolling(20,min_periods=15).std()
raw=-(R.sub(R.median(axis=1),axis=0).rolling(3,min_periods=3).sum()/vol); base=-R.rolling(5,min_periods=4).sum()/R.rolling(5,min_periods=4).std()
out=[]
for d in raw.index:
 q=pd.concat([raw.loc[d],base.loc[d]],axis=1).dropna(); x=raw.loc[d].copy()*np.nan
 if len(q)>=8 and np.var(q.iloc[:,1])>0:
  b=np.cov(q.iloc[:,0],q.iloc[:,1],ddof=0)[0,1]/np.var(q.iloc[:,1]);x=raw.loc[d]-b*base.loc[d]
 out.append(x)
f=pd.DataFrame(out,index=raw.index,columns=raw.columns)
print('candidate orthogonalized_cross_asset_reversal; dates',len(P),'assets',len(A),'last',P.index.max().date())
for h in [1,5,10,20]:
 z=[];ns=[];ds=[]; fut=P.pct_change(h).shift(-h)
 for d in P.index:
  q=pd.concat([f.loc[d],fut.loc[d]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q));ds.append(d)
 z=pd.Series(z,index=ds);print('H',h,'valid_dates',len(z),'avg_n',round(np.mean(ns),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4),'years',z.groupby(z.index.year).mean().round(6).to_dict())
print('coverage',round(f.notna().stack().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
for n,x in {'ravmom':R.rolling(20,min_periods=15).sum()/vol,'invvol':-vol,'shortrev':base,'peer':R.shift(2).sub(R.shift(2).mean(axis=1),axis=0)}.items():
 q=pd.concat([f.stack(),x.stack()],axis=1).dropna();print('library_rho',n,round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6),'cells',len(q))
