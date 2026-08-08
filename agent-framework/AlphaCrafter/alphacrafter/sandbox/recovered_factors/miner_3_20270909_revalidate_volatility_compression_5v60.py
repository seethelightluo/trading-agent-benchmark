"""miner_3 scheduled revalidation: short-to-long volatility compression."""
import pandas as pd,numpy as np,glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2027-09-08')
def read(a,field='close',base='../persistent/stock_data/'):
 d=pd.read_csv(base+a+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d[field],errors='coerce')
P=pd.DataFrame({a:read(a) for a in A});R=P.pct_change(fill_method=None);v5=R.rolling(5,min_periods=4).std();v20=R.rolling(20,min_periods=15).std();v60=R.rolling(60,min_periods=45).std();F=-v5/(v60+1e-12)
def csres(x,C):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for d in x.index:
  q=pd.concat([x.loc[d].rename('y')]+[z.loc[d].rename(str(i)) for i,z in enumerate(C)],axis=1).dropna(); X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:o.loc[d,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return o
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
def macro(n):
 z=read(n,base='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index);return csres(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(z) for a in A}),[peer])
vix=read('VIX',base='../persistent/index_data/').reindex(P.index);shock=(vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0)
ind=R.lt(0).astype(float); pp=ind.rolling(20,min_periods=12).mean(); loss=pd.DataFrame({a:ind[a].rolling(20,min_periods=12).cov(ind[a].shift(1))/(pp[a]*(1-pp[a])+1e-12) for a in A})
rav=(P/P.shift(20)-1)/v20;rev=-(P/P.shift(5)-1)/v5
H=pd.DataFrame({a:read(a,'high') for a in A});Lo=pd.DataFrame({a:read(a,'low') for a in A});loc=(P-Lo)/(H-Lo).replace(0,np.nan);cl=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A})
invup=-csres(R.where(R>0,0).rolling(20,min_periods=15).mean(),[rav]); invskew=csres(csres(-R.rolling(20,min_periods=15).skew(),[rav]),[v20]);M=R.mean(axis=1)
def bt(a,mask):
 x=M.where(mask);y=R[a].where(mask);return y.rolling(30,min_periods=10).cov(x)/x.rolling(30,min_periods=10).var()
dba=pd.DataFrame({a:-(bt(a,M<0)-bt(a,M>0)) for a in A})
LIB={'ravmom_20obs':rav,'volnorm_reversal_5obs':rev,'vix_beta_residual_peer20':macro('VIX'),'dxy_beta_residual_peer20':macro('DXY'),'downside_close_location_resilience_20':cl,'realized_volatility_20obs':v20,'peer_crowding_correlation_20obs':peer,'vix_conditioned_peer_crowding_20':peer.mul(shock,axis=0),'loss_clustering_20obs':loss,'inverse_ravmom_residual_upside_participation_20obs':invup,'inverse_residual_return_skewness_20obs':invskew,'low_volatility_of_volatility_20obs':-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean()),'downside_beta_asymmetry_30':dba}
print('REVALIDATION volatility_compression_5v60; visible through',E.date(),'assets',len(A),'comparators',len(LIB))
ICS={}; MET={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;out=[];ns=[]
 for d in P.index:
  q=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((d,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 x=pd.Series(dict(out));ICS[h]=x;sd=x.std(ddof=1);MET[h]=(x.mean(),x.mean()/sd,(x>0).mean(),len(x),np.mean(ns),sd/np.sqrt(len(x)));print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.6f} instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
for name,mask in [('2020_21',ICS[10].index<'2022-01-01'),('2022_23',(ICS[10].index>='2022-01-01')&(ICS[10].index<'2024-01-01')),('2024_25',(ICS[10].index>='2024-01-01')&(ICS[10].index<'2026-01-01')),('2026_27',ICS[10].index>='2026-01-01')]:
 x=ICS[10][mask];print('REGIME',name,'dates',len(x),'IC',None if len(x)==0 else f'{x.mean():.6f}','ICIR',None if len(x)<2 else f'{x.mean()/x.std(ddof=1):.6f}','hit',None if len(x)==0 else f'{(x>0).mean():.6f}')
r=F.rank(axis=1,pct=True);ts=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:ts.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'rank_turnover',f'{np.mean(ts):.6f}')
mx=0;who='';cells=0
for n,x in LIB.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman');print('library',n,f'rho={rho:.6f}','common_cells',len(q))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',cells);print('METRIC',MET)
