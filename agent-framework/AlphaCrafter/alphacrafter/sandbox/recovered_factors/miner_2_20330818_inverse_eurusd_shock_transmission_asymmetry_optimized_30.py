"""Miner_2: optimized validation of one EURUSD shock-asymmetry candidate, cutoff 2033-08-17."""
import pandas as pd, numpy as np
from pathlib import Path
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2033-08-17')
def rd(a,c='close',root='../persistent/stock_data/'):
 d=pd.read_csv(root+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return pd.to_numeric(d.loc[d.index<=E,c],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None); M=R.mean(1); v=R.rolling(20,min_periods=15).std(); v5=R.rolling(5,min_periods=4).std(); trend=P/P.shift(20)-1
def residual(x,*cs):
 y=x.to_numpy(float); zz=[c.to_numpy(float) for c in cs]; out=np.full(y.shape,np.nan)
 for i in range(len(P)):
  X=np.column_stack([z[i] for z in zz]); ok=np.isfinite(y[i])&np.isfinite(X).all(1)
  if ok.sum()>=8:
   xx=np.c_[np.ones(ok.sum()),X[ok]]
   if np.linalg.matrix_rank(xx)==xx.shape[1]: out[i,ok]=y[i,ok]-xx@np.linalg.lstsq(xx,y[i,ok],rcond=None)[0]
 return pd.DataFrame(out,index=P.index,columns=A)
def beta(z,mask,w=30,n=10):
 q=z.where(mask); den=q.rolling(w,min_periods=n).var()
 return pd.DataFrame({a:R[a].where(mask).rolling(w,min_periods=n).cov(q)/den for a in A})
peer=pd.DataFrame({a:pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(1) for a in A})
dba=-(beta(M,M<0)-beta(M,M>0))
# Candidate is the inverse positive-minus-negative EURUSD beta, purged of generic risk descriptors.
euro=rd('EURUSD',root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
F=-residual(beta(euro,euro>0)-beta(euro,euro<0),v,peer,dba,trend)
# Reconstruct every distinct active-library family with economical matching proxies.
def macro(n):
 z=rd(n,root='../persistent/index_data/').pct_change(fill_method=None).reindex(P.index)
 return residual(pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(z) for a in A}),peer)
I=(R<0).astype(float); loss=pd.DataFrame({a:I[a].rolling(20,min_periods=15).cov(I[a].shift())/(I[a].rolling(20,min_periods=15).mean()*(1-I[a].rolling(20,min_periods=15).mean())) for a in A})
mom=trend/(v+1e-12); rev=-(P/P.shift(5)-1)/(v5+1e-12); invup=-residual(R.where(R>0,0).rolling(20,min_periods=15).mean(),mom); invsk=residual(residual(-R.rolling(20,min_periods=15).skew(),mom),v)
vix=rd('VIX',root='../persistent/index_data/').reindex(P.index); shock=(vix/vix.rolling(60,min_periods=40).mean()-1).clip(lower=0)
post=R.where(R.shift(1)<0).rolling(20,min_periods=6).mean()/(v+1e-12); postres=residual(residual(post,loss),mom)
dd=P/P.rolling(20,min_periods=15).max()-1; stress=(-M.shift(1)/(M.shift(1).rolling(60,min_periods=45).std()+1e-12)).clip(0,4); w=(-dd.shift(1)/(v.shift(1)+1e-12)).clip(0,5).mul(stress,axis=0); sr=R.mul(w).rolling(20,min_periods=15).sum().div(w.rolling(20,min_periods=15).sum().replace(0,np.nan),axis=0)/(v+1e-12); stressrec=residual(residual(residual(residual(sr,post),loss),mom),dd/(v+1e-12))
O=pd.DataFrame({a:rd(a,'open') for a in A}); H=pd.DataFrame({a:rd(a,'high') for a in A}); Lo=pd.DataFrame({a:rd(a,'low') for a in A}); loc=(P-Lo)/(H-Lo).replace(0,np.nan); cl=pd.DataFrame({a:loc[a].where(R[a]<0).rolling(20,min_periods=6).mean()-loc[a].where(R[a]>0).rolling(20,min_periods=6).mean() for a in A})
L={'dxy_beta':macro('DXY'),'vix_beta':macro('VIX'),'ravmom':mom,'reversal':rev,'close_location':cl,'realized_vol':v,'peer_crowding':peer,'vix_crowding':peer.mul(shock,axis=0),'loss_cluster':loss,'inverse_upside':invup,'inverse_skew':invsk,'downside_beta':dba,'post_loss':postres,'stress_recovery':stressrec}
for nm,z,sgn in [('dxy','DXY',1),('oil','WTI',-1),('usdjpy','USDJPY',-1)]:
 q=rd(z,root='../persistent/stock_data/' if z=='WTI' else '../persistent/index_data/').pct_change(fill_method=None).reindex(P.index); L[nm+'_asym']=residual(sgn*(beta(q,q>0)-beta(q,q<0)),v,peer,dba,trend)
for nm,z in [('rate',R.US10Y),('crypto',R.BTC)]: L['downside_'+nm]=residual(beta(z,M<0)-beta(z,M>=0),v,peer,dba,trend)
idir=pd.DataFrame({a:(P[a]-O[a])/(H[a]-Lo[a]).replace(0,np.nan) for a in A}).rolling(20,min_periods=12).mean(); L['intraday_eff']= -residual(idir,trend,v,peer,dba)
print('FACTOR inverse_eurusd_shock_transmission_asymmetry_residual_30 cutoff',E.date(),'dates',len(P),'assets',len(A),'library_proxies',len(L))
ics={}
for h in [1,5,10,20]:
 fw=P.shift(-h)/P-1; xs=[]; nn=[]
 for t in P.index:
  q=pd.concat([F.loc[t].rename('f'),fw.loc[t].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1: xs.append((t,q.f.corr(q.r,method='spearman')));nn.append(len(q))
 x=pd.Series(dict(xs));ics[h]=x; print('h',h,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}','instruments',f'{np.mean(nn):.2f}','se',f'{x.std(ddof=1)/np.sqrt(len(x)):.6f}')
for label,mask in [('2020_24',ics[20].index<'2025-01-01'),('2025_28',(ics[20].index>='2025-01-01')&(ics[20].index<'2029-01-01')),('2029_now',ics[20].index>='2029-01-01')]:
 x=ics[20][mask];print('regime20',label,'dates',len(x),'IC',f'{x.mean():.6f}','ICIR',f'{x.mean()/x.std(ddof=1):.6f}','hit',f'{(x>0).mean():.4f}')
r=F.rank(axis=1,pct=True); tos=[]
for i in range(1,len(r)):
 q=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(q)>=8: tos.append(1-q.iloc[:,0].corr(q.iloc[:,1],method='spearman'))
print('coverage',f'{F.notna().mean().mean():.6f}','valid_cells',int(F.notna().sum().sum()),'turnover',f'{np.mean(tos):.6f}')
vals=[]
for n,x in L.items():
 q=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna(); rho=q.f.corr(q.x,method='spearman'); vals.append((abs(rho),n,rho,len(q)));print('library',n,'rho',f'{rho:.6f}','cells',len(q))
b=max(vals);print('max_abs_library_correlation',f'{b[0]:.6f}','against',b[1],'cells',b[3])
