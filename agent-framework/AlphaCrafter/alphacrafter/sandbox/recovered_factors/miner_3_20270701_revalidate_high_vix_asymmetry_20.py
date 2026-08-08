"""Miner_3 scheduled revalidation: high-VIX momentum-residual downside asymmetry."""
import pandas as pd,numpy as np,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2027-06-30')
def rd(a,field='close',root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d[field],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None); v5=R.rolling(5,min_periods=4).std();v20=R.rolling(20,min_periods=15).std();v60=R.rolling(60,min_periods=45).std()
def csres(y, controls):
 o=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for d in P.index:
  q=pd.concat([y.loc[d].rename('y')]+[x.loc[d].rename(str(i)) for i,x in enumerate(controls)],axis=1).dropna(); X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:o.loc[d,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return o
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
def macro(n):
 m=rd(n,root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
 return csres(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in A}),[peer])
vix=rd('VIX',root='../persistent/index_data/').reindex(P.index); shock=(vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0)
up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
F=csres(np.log((up+1e-8)/(dn+1e-8)),[P/P.shift(20)-1]).mul(shock,axis=0)
# all other currently admitted signals reconstructed at signal-cell level
M=R.mean(axis=1)
def dbeta(a,mask):
 x=M.where(mask);y=R[a].where(mask);return y.rolling(30,min_periods=10).cov(x)/x.rolling(30,min_periods=10).var()
downbeta=pd.DataFrame({a:-(dbeta(a,M<0)-dbeta(a,M>0)) for a in A})
loss=pd.DataFrame({a:R[a].lt(0).astype(float).rolling(20,min_periods=15).cov(R[a].lt(0).astype(float).shift(1))/((R[a].lt(0).astype(float).rolling(20,min_periods=15).mean())*(1-R[a].lt(0).astype(float).rolling(20,min_periods=15).mean())) for a in A})
H=pd.DataFrame({a:rd(a,'high') for a in A});Lo=pd.DataFrame({a:rd(a,'low') for a in A});loc=(P-Lo)/(H-Lo).replace(0,np.nan);dcl=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A})
rav=(P/P.shift(20)-1)/v20; rev=-(P/P.shift(5)-1)/v5; invup=-csres(R.where(R>0,0).rolling(20,min_periods=15).mean(),[rav]); invskew=csres(csres(-R.rolling(20,min_periods=15).skew(),[rav]),[v20])
L={'ravmom':rav,'volnorm_reversal':rev,'vix_beta_residual':macro('VIX'),'dxy_beta_residual':macro('DXY'),'downside_close_location':dcl,'realized_vol':v20,'peer_crowding':peer,'vix_conditioned_crowding':peer.mul(shock,axis=0),'loss_clustering':loss,'inverse_rav_upside':invup,'inverse_residual_skew':invskew,'low_vol_of_vol':-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean()),'vol_compression':-(v5/(v60+1e-12)),'downside_beta_asymmetry':downbeta}
print('REVALIDATION high_vix_momentum_residual_downside_asymmetry_20; visible through',E.date(),'assets',len(A),'comparators',len(L))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;out=[];ns=[]
 for d in P.index:
  q=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((d,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;sd=x.std(ddof=1);print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
for label,mask in [('2020_21',ics[10].index<'2022-01-01'),('2022_23',(ics[10].index>='2022-01-01')&(ics[10].index<'2024-01-01')),('2024_25',(ics[10].index>='2024-01-01')&(ics[10].index<'2026-01-01')),('2026_27',ics[10].index>='2026-01-01')]:
 x=ics[10][mask];print(label,'dates',len(x),'IC',None if len(x)==0 else f'{x.mean():.6f}','ICIR',None if len(x)<2 else f'{x.mean()/x.std(ddof=1):.6f}','hit',None if len(x)==0 else f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True);turn=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:turn.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','rank_turnover',f'{np.mean(turn):.6f}')
mx=0;who='';cells=0
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman');print('library',n,f'rho={rho:.6f}','common_cells',len(q))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',cells)
