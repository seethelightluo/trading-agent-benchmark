import pandas as pd,numpy as np,glob,json,os
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2033-12-07')
def rd(a,c='close'):
 p='../persistent/stock_data/'+a+'.csv'; d=pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}).loc['2020-01-01':]; O=pd.DataFrame({a:rd(a,'open') for a in A}).reindex(P.index); R=P.pct_change(fill_method=None); intr=P/O-1; overnight=O/P.shift(1)-1
v=R.rolling(20,min_periods=15).std(); trend=P/P.shift(20)-1; M=R.mean(1); peer=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(R.drop(columns=a).mean(1)) for a in A})
def res(x,cs):
 out=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for t in x.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(i)) for i,z in enumerate(cs)],axis=1).dropna();
  if len(q)>=8:
   X=np.c_[np.ones(len(q)),q.iloc[:,1:]]
   if np.linalg.matrix_rank(X)==X.shape[1]: out.loc[t,q.index]=q.y-X@np.linalg.lstsq(X,q.y,rcond=None)[0]
 return out
# Overnight strength versus intraday strength, smoothed to reduce noise and residualized.
F=res((intr.rolling(20,min_periods=15).mean()-overnight.rolling(20,min_periods=15).mean())/(v+1e-12),[v,peer,trend])
print('candidate=intraday_overnight_rotation_residual_20')
print('rows',len(P),'dates',F.notna().any(1).sum(),'mean_n',F.notna().sum(1).replace(0,np.nan).mean(),'coverage',F.notna().mean().mean())
for h in [1,5,10,20]:
 ic=[]; turn=[]
 for i in range(len(P)-h):
  q=pd.concat([F.iloc[i].rename('f'),R.shift(-h).iloc[i].rename('r')],axis=1).dropna()
  if len(q)>=8: ic.append(q.f.corr(q.r,method='spearman'))
 print('h',h,'IC',np.nanmean(ic),'ICIR',np.nanmean(ic)/(np.nanstd(ic,ddof=1)+1e-12),'hit',np.mean(np.array(ic)>0),'n',len(ic))
print('turnover',F.rank(axis=1,pct=True).diff().abs().mean().mean())
# independence screen against simple library proxies plus persisted definitions where reconstructable
proxies={'mom':trend/(v+1e-12),'rev':-P.pct_change(5)/(R.rolling(5).std()+1e-12),'vol':v,'peer':peer,'intraday':intr.rolling(20,min_periods=15).mean()/(v+1e-12),'overnight':overnight.rolling(20,min_periods=15).mean()/(v+1e-12)}
mx=0; arg='none'
for n,z in proxies.items():
 c=[]
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),z.loc[t].rename('z')],axis=1).dropna()
  if len(q)>=8:c.append(q.f.corr(q.z,method='spearman'))
 if c and abs(np.nanmean(c))>mx:mx=abs(np.nanmean(c));arg=n
print('max_abs_library_correlation_proxy',mx,arg)
