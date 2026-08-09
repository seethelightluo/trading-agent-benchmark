"""Point-in-time fast revalidation of admitted inverse dispersion rate-spread residual."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2034-08-16')
def rd(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=E,'close'],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A});R=P.pct_change(fill_method=None);v=R.rolling(20,min_periods=15).std(); M=R.mean(1)
def beta(z,mask,w=30,n=10): return pd.DataFrame({a:R[a].where(mask).rolling(w,min_periods=n).cov(z.where(mask))/z.where(mask).rolling(w,min_periods=n).var() for a in A})
def res(x,*cs):
 out=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(i)) for i,z in enumerate(cs)],axis=1).dropna();X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:out.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return out
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A}); dba=-(beta(M,M<0)-beta(M,M>0));trend=P/P.shift(20)-1
sp=R.US10Y-R.CN10Y;spz=sp/(sp.rolling(60,min_periods=40).std()+1e-12);disp=R.std(axis=1);dz=((disp-disp.rolling(60,min_periods=40).mean())/(disp.rolling(60,min_periods=40).std()+1e-12)).clip(0,3);fx=spz*(1+dz.shift(1));F=-res(beta(fx,fx.notna(),30,15),v,peer,dba,trend)
print('cutoff',E.date(),'rows',len(P),'assets',len(A),'coverage',F.notna().mean().mean(),'cells',int(F.notna().sum().sum()))
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;vals=[];ns=[];ds=[]
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:vals.append(q.f.corr(q.r,method='spearman'));ns.append(len(q));ds.append(t)
 x=pd.Series(vals);print('h',h,'dates',len(x),'meanN',np.mean(ns),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean())
 if h==20:
  for lo,hi in [('2020','2025'),('2025','2030'),('2030','2035')]:
   y=x[[lo<=str(d.year)<hi for d in ds]];print('regime',lo,hi,'dates',len(y),'IC',y.mean(),'ICIR',y.mean()/y.std(ddof=1))
r=F.rank(axis=1,pct=True);tos=[]
for j in range(1,len(r)):
 q=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(q)>=8:tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('turnover',np.nanmean(tos),'turnover_dates',len(tos))
print('library_audit NOT COMPUTED')
