"""Miner_3: idiosyncratic drawdown recovery persistence residual, visible 2028-10-18."""
import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2028-10-18')
def rd(a,c='close',root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A});R=P.pct_change(fill_method=None);M=R.mean(1);v=R.rolling(20,min_periods=15).std();dd=P/P.rolling(20,min_periods=15).max()-1
def res(x,*cs):
 out=pd.DataFrame(index=P.index,columns=A,dtype=float)
 for t in P.index:
  q=pd.concat([x.loc[t].rename('y')]+[z.loc[t].rename(str(i)) for i,z in enumerate(cs)],axis=1).dropna(); X=q.iloc[:,1:]
  if len(q)>=8 and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1: out.loc[t,q.index]=q.y-np.c_[np.ones(len(q)),X]@np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0]
 return out
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})
def beta(mask,w=30,n=10):
 m=M.where(mask);return pd.DataFrame({a:R[a].where(mask).rolling(w,min_periods=n).cov(m)/m.rolling(w,min_periods=n).var() for a in A})
dba=-(beta(M<0)-beta(M>0)); mom=(P/P.shift(20)-1)/(v+1e-12)
# High score: relative return after a prior idiosyncratic (not market-wide) loss is persistently positive.
rel=R.sub(M,axis=0); shock=rel.shift(1).lt(-0.75*v.shift(1))
raw=pd.DataFrame({a:rel[a].where(shock[a]).rolling(30,min_periods=8).mean()/(v[a]+1e-12) for a in A})
post=R.where(R.shift(1)<0).rolling(20,min_periods=6).mean()/(v+1e-12)
F=res(raw,post,dba,mom,v,peer)
# Reconstruct admitted library signals for full mandatory signal-correlation screen.
def macro(n):
 z=rd(n,root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index); return res(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(z) for a in A}),peer)
I=(R<0).astype(float);loss=pd.DataFrame({a:I[a].rolling(20,min_periods=15).cov(I[a].shift())/(I[a].rolling(20,min_periods=15).mean()*(1-I[a].rolling(20,min_periods=15).mean())) for a in A})
v5=R.rolling(5,min_periods=4).std(); rev=-(P/P.shift(5)-1)/(v5+1e-12); invup=-res(R.where(R>0,0).rolling(20,min_periods=15).mean(),mom); invsk=res(res(-R.rolling(20,min_periods=15).skew(),mom),v)
vix=rd('VIX',root='../persistent/index_data/').reindex(P.index);vk=(vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0); postres=res(res(post,loss),mom)
ddac=pd.DataFrame({a:-R[a].where(dd[a]<0).rolling(30,min_periods=12).corr(R[a].where(dd[a]<0).shift()) for a in A});upbeta=res(beta(M>0,w=20,n=8),mom,v,peer)
cl=pd.DataFrame({a:((P[a]-rd(a,'low'))/(rd(a,'high')-rd(a,'low')).replace(0,np.nan)).where(R[a]<0).rolling(20,min_periods=6).mean()-((P[a]-rd(a,'low'))/(rd(a,'high')-rd(a,'low')).replace(0,np.nan)).where(R[a]>0).rolling(20,min_periods=6).mean() for a in A})
L={'dxy_beta_residual_peer20':macro('DXY'),'vix_beta_residual_peer20':macro('VIX'),'ravmom_20obs':mom,'volnorm_reversal_5obs':rev,'downside_close_location_resilience_20':cl,'realized_volatility_20obs':v,'peer_crowding_correlation_20obs':peer,'vix_conditioned_peer_crowding_20':peer.mul(vk,axis=0),'loss_clustering_20obs':loss,'inverse_ravmom_residual_upside_participation_20obs':invup,'inverse_residual_return_skewness_20obs':invsk,'downside_beta_asymmetry_30':dba,'drawdown_conditioned_inverse_autocorrelation_30':ddac,'residual_post_loss_rebound_20':postres,'upside_market_capture_residual_20':upbeta}
print('FACTOR idiosyncratic_drawdown_recovery_persistence_residual_30 visible_through',E.date(),'assets',len(A),'library_signals',len(L));ics={}
for h in [1,5,10,20]:
 out=[];nn=[];fw=P.shift(-h)/P-1
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((t,q.f.corr(q.r,method='spearman')));nn.append(len(q))
 x=pd.Series(dict(out),dtype=float);ics[h]=x;sd=x.std(ddof=1);print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(nn):.2f} se={sd/np.sqrt(len(x)):.6f}')
for n,m in [('2020_21',ics[10].index<'2022-01-01'),('2022_23',(ics[10].index>='2022-01-01')&(ics[10].index<'2024-01-01')),('2024_25',(ics[10].index>='2024-01-01')&(ics[10].index<'2026-01-01')),('2026_27',(ics[10].index>='2026-01-01')&(ics[10].index<'2028-01-01')),('2028_ytd',ics[10].index>='2028-01-01')]:
 x=ics[10][m];print('regime',n,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}' if len(x)>1 else 'nan','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True);to=[]
for j in range(1,len(r)):
 q=pd.concat([r.iloc[j-1],r.iloc[j]],axis=1).dropna()
 if len(q)>=8:to.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(to):.6f}')
mx=-1
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman');print('library',n,f'rho={rho:.6f}','common_cells',len(q))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('max_abs_library_correlation',f'{mx:.6f}','against',who,'common_cells',cells)
