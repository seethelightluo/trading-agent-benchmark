import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2032-01-07')
def rd(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None); M=R.mean(axis=1); v=R.rolling(20,min_periods=15).std(); dd=P/P.rolling(20,min_periods=15).max()-1
O=pd.DataFrame({a:rd(a,'open') for a in A});H=pd.DataFrame({a:rd(a,'high') for a in A});Lo=pd.DataFrame({a:rd(a,'low') for a in A});loc=((P-Lo)/(H-Lo).replace(0,np.nan)).clip(0,1)
peer=pd.DataFrame({a:sum(R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a)/14 for a in A})
def bet(mask):
 m=M.where(mask); return pd.DataFrame({a:R[a].where(mask).rolling(30,min_periods=10).cov(m)/m.rolling(30,min_periods=10).var() for a in A})
dba=-(bet(M<0)-bet(M>0)); raw=(-(O/P.shift(1)-1)*(2*loc-1)).rolling(20,min_periods=12).mean()/(v+1e-12); cl=loc.where(R<0).rolling(20,min_periods=6).mean()-loc.where(R>0).rolling(20,min_periods=6).mean()
# cross-sectional OLS residual
F=pd.DataFrame(index=P.index,columns=A,dtype=float); trend=P/P.shift(20)-1
for t in P.index:
 q=pd.concat([raw.loc[t].rename('y'),v.loc[t].rename('v'),peer.loc[t].rename('p'),dba.loc[t].rename('b'),trend.loc[t].rename('t'),cl.loc[t].rename('c')],axis=1).dropna()
 if len(q)>=8:
  X=np.c_[np.ones(len(q)),q.iloc[:,1:].to_numpy()]
  if np.linalg.matrix_rank(X)==X.shape[1]: F.loc[t,q.index]=q.y.to_numpy()-X@np.linalg.lstsq(X,q.y.to_numpy(),rcond=None)[0]
print('candidate gap_fade_efficiency_residual_20 cutoff',E.date(),'assets',len(A),flush=True)
ics={}
for h in (1,5,10,20):
 fw=pd.DataFrame({a:(P[a].dropna().shift(-h)/P[a].dropna()-1).reindex(P.index) for a in A}); x=[]; ns=[]
 for t in P.index:
  q=pd.concat([F.loc[t],fw.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:x.append((t,q.iloc[:,0].corr(q.iloc[:,1],method='spearman')));ns.append(len(q))
 z=pd.Series(dict(x));ics[h]=z; sd=z.std(ddof=1); print('h',h,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/sd,6),'hit',round((z>0).mean(),4),'instruments',round(np.mean(ns),2),'se',round(sd/np.sqrt(len(z)),6))
z=ics[10]
for n,l,u in [('2020_21','2020-01-01','2022-01-01'),('2022_23','2022-01-01','2024-01-01'),('2024_25','2024-01-01','2026-01-01'),('2026_27','2026-01-01','2028-01-01'),('2028_32','2028-01-01','2033-01-01')]:
 x=z[(z.index>=l)&(z.index<u)];print('regime',n,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round((x>0).mean(),4))
r=F.rank(axis=1,pct=True);tos=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',round(F.notna().mean().mean(),6),'cells',int(F.notna().sum().sum()),'turnover',round(np.mean(tos),6))
