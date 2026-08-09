"""miner_3 scheduled revalidation: low volatility-of-volatility, 2027-11-18."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2027-11-17')
def read(a,field='close',base='../persistent/stock_data/'):
 d=pd.read_csv(base+a+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d[field],errors='coerce')
P=pd.DataFrame({a:read(a) for a in A});R=P.pct_change(fill_method=None);v5=R.rolling(5,min_periods=4).std();v20=R.rolling(20,min_periods=15).std();v60=R.rolling(60,min_periods=45).std();F=-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean())
def csres(x,controls):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for d in x.index:
  q=pd.concat([x.loc[d].rename('y')]+[z.loc[d].rename(str(i)) for i,z in enumerate(controls)],axis=1).dropna();X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:o.loc[d,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return o
PE=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
def macro(n):
 z=read(n,base='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index);return csres(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(z) for a in A}),[PE])
vix=read('VIX',base='../persistent/index_data/').reindex(P.index);shock=(vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0);rav=(P/P.shift(20)-1)/v20;rev=-(P/P.shift(5)-1)/v5
H=pd.DataFrame({a:read(a,'high') for a in A});Lo=pd.DataFrame({a:read(a,'low') for a in A});loc=(P-Lo)/(H-Lo).replace(0,np.nan);dcl=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A})
ind=R.lt(0).astype(float);pn=ind.rolling(20,min_periods=15).mean();loss=pd.DataFrame({a:ind[a].rolling(20,min_periods=15).cov(ind[a].shift(1))/(pn[a]*(1-pn[a])+1e-12) for a in A});invup=-csres(R.where(R>0,0).rolling(20,min_periods=15).mean(),[rav]);invsk=csres(csres(-R.rolling(20,min_periods=15).skew(),[rav]),[v20]);M=R.mean(axis=1)
def beta(a,m):
 x=M.where(m);y=R[a].where(m);return y.rolling(30,min_periods=10).cov(x)/x.rolling(30,min_periods=10).var()
db=pd.DataFrame({a:-(beta(a,M<0)-beta(a,M>0)) for a in A});under=P/P.rolling(20,min_periods=15).max()-1;dd=-pd.DataFrame({a:R[a].where(under[a]<0).rolling(30,min_periods=12).corr(R[a].where(under[a]<0).shift(1)) for a in A})
# All admitted signals except candidate itself.
L={'ravmom':rav,'reversal':rev,'vix_resid':macro('VIX'),'dxy_resid':macro('DXY'),'downside_close_location':dcl,'realized_vol':v20,'peer_crowding':PE,'vix_peer':PE.mul(shock,axis=0),'loss_clustering':loss,'inverse_upside':invup,'inverse_skew':invsk,'vol_compression':-(v5/(v60+1e-12)),'downside_beta':db,'drawdown_inverse_ac':dd}
print('REVALIDATION low_vov; visible through',E.date(),'assets',len(A),'comparators',len(L))
ics={}; metrics={}
for h in [1,5,10,20]:
 out=[];ns=[];fw=P.shift(-h)/P-1
 for d in P.index:
  q=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((d,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x; sd=x.std(ddof=1);metrics[h]=(x.mean(),x.mean()/sd,(x>0).mean(),len(x),np.mean(ns),sd/np.sqrt(len(x)));print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
for name,ma in [('2026',ics[10].index<'2027-01-01'),('2027',ics[10].index>='2027-01-01')]:
 x=ics[10][ma];print(name,'10d_dates',len(x),'IC',f'{x.mean():.6f}' if len(x) else None,'ICIR',f'{x.mean()/x.std(ddof=1):.6f}' if len(x)>1 else None)
r=F.rank(axis=1,pct=True);to=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'rank_turnover',f'{np.mean(to):.6f}')
mx=0
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();z=q.f.corr(q.x,method='spearman');print('library',n,f'rho={z:.6f}','cells',len(q))
 if abs(z)>mx:mx=abs(z);who=n;cells=len(q)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'cells',cells);print('METRIC',metrics)
