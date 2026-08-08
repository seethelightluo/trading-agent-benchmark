"""Miner_2 single-candidate research: inverse crypto-dispersion-shock beta residual (30 sessions)."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2033-06-23')
def rd(a,c='close',root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A});R=P.pct_change(fill_method=None);M=R.mean(1);v=R.rolling(20,min_periods=15).std();v5=R.rolling(5,min_periods=4).std()
def res(x,*cs):
 out=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(i)) for i,z in enumerate(cs)],axis=1).dropna();X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:out.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return out
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})
def beta(z,mask,w=30,n=10): return pd.DataFrame({a:R[a].where(mask).rolling(w,min_periods=n).cov(z.where(mask))/z.where(mask).rolling(w,min_periods=n).var() for a in A})
dba=-(beta(M,M<0)-beta(M,M>0));trend=P/P.shift(20)-1
# Pre-specified: exceptional BTC-vs-ETH return divergence denotes a crypto-specific
# relative-value shock. Assets with lower trailing shock beta are preferred, after controls.
cd=(R.BTC-R.ETH).abs();driver=(cd-cd.rolling(60,min_periods=40).mean())/(cd.rolling(60,min_periods=40).std()+1e-12)
F=res(-beta(driver,driver.notna(),30,15),v,peer,dba,trend)
# Reconstruct every current/admitted historical library family for a mandatory novelty check.
def macro(n):
 z=rd(n,root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index);return res(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(z) for a in A}),peer)
I=(R<0).astype(float);loss=pd.DataFrame({a:I[a].rolling(20,min_periods=15).cov(I[a].shift())/(I[a].rolling(20,min_periods=15).mean()*(1-I[a].rolling(20,min_periods=15).mean())) for a in A})
mom=trend/(v+1e-12);rev=-(P/P.shift(5)-1)/(v5+1e-12);invup=-res(R.where(R>0,0).rolling(20,min_periods=15).mean(),mom);invsk=res(res(-R.rolling(20,min_periods=15).skew(),mom),v)
vix=rd('VIX',root='../persistent/index_data/').reindex(P.index);shock=(vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0);post=R.where(R.shift(1)<0).rolling(20,min_periods=6).mean()/(v+1e-12);postres=res(res(post,loss),mom)
dd=P/P.rolling(20,min_periods=15).max()-1;strm=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,4);w=(-dd.shift(1)/(v.shift(1)+1e-12)).clip(0,5).mul(strm,axis=0);sr=R.mul(w).rolling(20,min_periods=15).sum().div(w.rolling(20,min_periods=15).sum().replace(0,np.nan),axis=0)/(v+1e-12);stressrec=res(res(res(res(sr,post),loss),mom),dd/(v+1e-12))
sev=(-R.shift(1)/(v.shift(1)+1e-12)).clip(0,5);cont=R.mul(sev).rolling(20,min_periods=10).sum().div(sev.rolling(20,min_periods=10).sum().replace(0,np.nan),axis=0)/(v+1e-12);contrec=res(res(res(cont,post),loss),mom);ddac=pd.DataFrame({a:-R[a].where(dd[a]<0).rolling(30,min_periods=12).corr(R[a].where(dd[a]<0).shift()) for a in A});upbeta=res(beta(M,M>0,20,8),mom,v,peer)
O=pd.DataFrame({a:rd(a,'open') for a in A});H=pd.DataFrame({a:rd(a,'high') for a in A});Lo=pd.DataFrame({a:rd(a,'low') for a in A});loc=(P-Lo)/(H-Lo).replace(0,np.nan);cl=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A})
L={'dxy_beta_residual_peer20':macro('DXY'),'vix_beta_residual_peer20':macro('VIX'),'ravmom_20obs':mom,'volnorm_reversal_5obs':rev,'downside_close_location_resilience_20':cl,'realized_volatility_20obs':v,'peer_crowding_correlation_20obs':peer,'vix_conditioned_peer_crowding_20':peer.mul(shock,axis=0),'loss_clustering_20obs':loss,'inverse_ravmom_residual_upside_participation_20obs':invup,'inverse_residual_return_skewness_20obs':invsk,'downside_beta_asymmetry_30':dba,'drawdown_conditioned_inverse_autocorrelation_30':ddac,'residual_post_loss_rebound_20':postres,'drawdown_conditioned_market_stress_recovery_residual_20':stressrec,'continuous_loss_severity_recovery_residual_20':contrec,'upside_market_capture_residual_20':upbeta}
for nm,z,orient in [('dxy','DXY',1),('oil','WTI',-1),('usdjpy','USDJPY',-1)]:
 q=rd(z,root=('../persistent/stock_data/' if z=='WTI' else '../persistent/index_data/')).pct_change(fill_method=None).reindex(P.index);L[nm+'_shock_asymmetry']=res(orient*(beta(q,q>0)-beta(q,q<0)),v,peer,dba,trend)
for nm,z in [('yield',R.US10Y),('crypto',R.BTC)]:L['downside_'+nm+'_transmission']=res(beta(z,M<0)-beta(z,M>=0),v,peer,dba,trend)
z=rd('VIX',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index);z=z/(z.rolling(60,min_periods=40).std()+1e-12);L['vix_surprise_transmission']=res(-(beta(z,z>=z.rolling(60,min_periods=40).median())-beta(z,z<z.rolling(60,min_periods=40).median())),v,peer,dba,trend)
idir=pd.DataFrame({a:(P[a]-O[a])/(H[a]-Lo[a]).replace(0,np.nan) for a in A}).rolling(20,min_periods=12).mean();L['intraday_directional_efficiency']=-res(idir,trend,v,peer,dba);dz=(R.std(axis=1)-R.std(axis=1).rolling(60,min_periods=40).mean())/(R.std(axis=1).rolling(60,min_periods=40).std()+1e-12);L['dispersion_conditioned_reversal']=res((-(P/P.shift(10)-1)/(v+1e-12)).mul(1+dz.clip(0,3),axis=0),trend,v,peer,dba)
print('FACTOR inverse_crypto_dispersion_shock_transmission_residual_30 visible_through',P.index.max().date(),'assets',len(A),'library_signals',len(L));ics={}
for h in [1,5,10,20]:
 out=[];ns=[];fw=P.shift(-h)/P-1
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((t,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={x.std(ddof=1)/np.sqrt(len(x)):.6f}')
for n,m in [('2026_27',(ics[20].index>='2026-01-01')&(ics[20].index<'2028-01-01')),('2028_current',ics[20].index>='2028-01-01')]:
 x=ics[20][m];print('regime20',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True);tos=[]
for j in range(1,len(r)):
 q=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(q)>=8:tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(tos):.6f}')
vals=[]
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman');vals.append((abs(rho),n,rho,len(q)));print('library',n,f'rho={rho:.6f}','common_cells',len(q))
best=max(vals);print('max_abs_library_correlation',f'{best[0]:.6f}','against',best[1],'common_cells',best[3])
