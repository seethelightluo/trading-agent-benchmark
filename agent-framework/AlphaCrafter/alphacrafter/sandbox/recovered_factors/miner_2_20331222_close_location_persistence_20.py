import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2033-12-21')
def rd(a,c):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
C=pd.DataFrame({a:rd(a,'close') for a in A}).loc['2020-01-01':]; H=pd.DataFrame({a:rd(a,'high') for a in A}).reindex(C.index); L=pd.DataFrame({a:rd(a,'low') for a in A}).reindex(C.index); O=pd.DataFrame({a:rd(a,'open') for a in A}).reindex(C.index)
R=C.pct_change(fill_method=None); rng=(H-L).replace(0,np.nan); cl=((C-L)/rng-.5); # signed close-location, positive resilient closes
F=cl.rolling(20,min_periods=15).mean()/(R.rolling(20,min_periods=15).std()+1e-12)
# remove common trend/volatility cross-sectional confounds each date
trend=C/C.shift(20)-1; vol=R.rolling(20,min_periods=15).std()
def res(x,cs):
 out=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for t in x.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(i)) for i,z in enumerate(cs)],axis=1).dropna()
  if len(q)>=8:
   X=np.c_[np.ones(len(q)),q.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]: out.loc[t,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
 return out
F=res(F,[trend,vol])
print('candidate=close_location_persistence_20'); print('rows',len(C),'dates',F.notna().any(1).sum(),'mean_n',F.notna().sum(1).replace(0,np.nan).mean(),'coverage',F.notna().mean().mean())
for h in [1,5,10,20]:
 ic=[]
 for i in range(len(C)-h):
  q=pd.concat([F.iloc[i].rename('f'),R.shift(-h).iloc[i].rename('r')],axis=1).dropna()
  if len(q)>=8: ic.append(q.f.corr(q.r,method='spearman'))
 a=np.array(ic); print('h',h,'IC',np.nanmean(a),'ICIR',np.nanmean(a)/(np.nanstd(a,ddof=1)+1e-12),'hit',np.mean(a>0),'n',len(a))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
# proxy library independence screen, conservative maximum
proxies={'trend':trend,'vol':vol,'close_location':cl.rolling(20,min_periods=15).mean(),'range':rng.rolling(20,min_periods=15).mean()/C.rolling(20).mean()}
mx=0; arg='none'
for n,z in proxies.items():
 vals=[]
 for t in C.index:
  q=pd.concat([F.loc[t].rename('f'),z.loc[t].rename('z')],axis=1).dropna()
  if len(q)>=8: vals.append(q.f.corr(q.z,method='spearman'))
 if vals and abs(np.nanmean(vals))>mx: mx=abs(np.nanmean(vals));arg=n
print('proxy_max_abs_mean_spearman',mx,arg)
