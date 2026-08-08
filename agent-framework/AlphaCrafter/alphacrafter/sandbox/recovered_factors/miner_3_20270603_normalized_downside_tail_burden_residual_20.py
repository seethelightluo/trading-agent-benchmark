"""miner_3: test normalized downside-tail burden, cross-sectionally residualized from known risk signals."""
import pandas as pd, numpy as np, glob
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2027-06-02')
def col(a,c='close'):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d[c],errors='coerce')
def idx(n):
 d=pd.read_csv('../persistent/index_data/'+n+'.csv',parse_dates=['date']).query('date<=@E').drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.close,errors='coerce')
P=pd.DataFrame({a:col(a) for a in A}); R=P.pct_change(fill_method=None); M=R.mean(axis=1); v5=R.rolling(5,min_periods=4).std(); v20=R.rolling(20,min_periods=15).std();v60=R.rolling(60,min_periods=45).std()
def csres(x, controls):
 o=pd.DataFrame(index=x.index,columns=A,dtype=float)
 for d in x.index:
  q=pd.concat([x.loc[d].rename('y')]+[z.loc[d].rename(str(k)) for k,z in enumerate(controls)],axis=1).dropna()
  X=q.iloc[:,1:]
  if len(q)>=8 and X.shape[1]<len(q) and np.linalg.matrix_rank(np.c_[np.ones(len(q)),X])==X.shape[1]+1:
   b=np.linalg.lstsq(np.c_[np.ones(len(q)),X],q.y,rcond=None)[0];o.loc[d,q.index]=q.y-np.c_[np.ones(len(q)),X]@b
 return o
def peer(): return pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1) for a in A})
PE=peer()
def dbeta():
 def b(a,mask):
  x=M.where(mask);y=R[a].where(mask);return y.rolling(30,min_periods=10).cov(x)/x.rolling(30,min_periods=10).var()
 return pd.DataFrame({a:-(b(a,M<0)-b(a,M>0)) for a in A})
DB=dbeta(); loss=pd.DataFrame({a:R[a].lt(0).astype(float).rolling(20,min_periods=15).cov(R[a].lt(0).astype(float).shift(1))/((R[a].lt(0).astype(float).rolling(20,min_periods=15).mean())*(1-R[a].lt(0).astype(float).rolling(20,min_periods=15).mean())) for a in A})
# One idea: tail-loss severity relative to each asset's own 20d volatility, removing known volatility/crowding/loss-clustering/downside-beta effects.
tail=-((-R).div(v20).where(R.lt(-v20))).rolling(20,min_periods=15).mean()
F=csres(tail,[v20,PE,loss,DB])
# Recreate all admitted signals for mandatory complete correlation evidence.
vix=idx('VIX').reindex(P.index); shock=(vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0)
up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
rav=(P/P.shift(20)-1)/v20; rev=-(P/P.shift(5)-1)/v5
high=csres(np.log((up+1e-8)/(dn+1e-8)),[P/P.shift(20)-1]).mul(shock,axis=0)
lowvov=-(v5.rolling(20,min_periods=15).std()/v5.rolling(20,min_periods=15).mean()); comp=-(v5/(v60+1e-12))
def macro(n):
 z=idx(n).reindex(P.index).pct_change(fill_method=None);x=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(z) for a in A});return csres(x,[PE])
# close-location resilience
H=pd.DataFrame({a:col(a,'high') for a in A});Lo=pd.DataFrame({a:col(a,'low') for a in A}); loc=(P-Lo)/(H-Lo).replace(0,np.nan)
dcl=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A})
up_part=R.where(R>0,0).rolling(20,min_periods=15).mean(); invup=-csres(up_part,[rav]); skew=-R.rolling(20,min_periods=15).skew(); invskew=csres(csres(skew,[rav]),[v20])
L={'ravmom_20obs':rav,'volnorm_reversal_5obs':rev,'vix_beta_residual_peer20':macro('VIX'),'dxy_beta_residual_peer20':macro('DXY'),'downside_close_location_resilience_20':dcl,'realized_volatility_20obs':v20,'peer_crowding_correlation_20obs':PE,'vix_conditioned_peer_crowding_20':PE.mul(shock,axis=0),'loss_clustering_20obs':loss,'inverse_ravmom_residual_upside_participation_20obs':invup,'inverse_residual_return_skewness_20obs':invskew,'high_vix_momentum_residual_downside_asymmetry_20':high,'low_volatility_of_volatility_20obs':lowvov,'volatility_compression_5v60':comp,'downside_beta_asymmetry_30':DB}
print('FACTOR normalized_downside_tail_burden_residual_20; visible through',E.date(),'assets',len(A),'library records',len(L))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1;out=[];ns=[]
 for d in P.index:
  q=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:out.append((d,q.f.corr(q.r,method='spearman')));ns.append(len(q))
 x=pd.Series(dict(out));ics[h]=x;sd=x.std(ddof=1);print(f'h={h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} instruments={np.mean(ns):.2f} se={sd/np.sqrt(len(x)):.6f}')
for label,mask in [('2020_21',ics[10].index<'2022-01-01'),('2022_23',(ics[10].index>='2022-01-01')&(ics[10].index<'2024-01-01')),('2024_25',(ics[10].index>='2024-01-01')&(ics[10].index<'2026-01-01')),('2026_27',ics[10].index>='2026-01-01')]:
 x=ics[10][mask];print(label,'dates',len(x),'IC',None if len(x)==0 else f'{x.mean():.6f}','ICIR',None if len(x)<2 else f'{x.mean()/x.std(ddof=1):.6f}','hit',None if len(x)==0 else f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True); turns=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8:turns.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','rank_turnover',f'{np.mean(turns):.6f}')
mx=-1;who=''
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=q.f.corr(q.x,method='spearman');print('library',n,f'rho={rho:.6f}','common_cells',len(q))
 if abs(rho)>mx:mx=abs(rho);who=n
print('max_abs_library_correlation',f'{mx:.6f}','against',who)
