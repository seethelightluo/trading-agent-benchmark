"""miner_3 one idea: VIX-shock-gated residual 5d reversal.
A short-horizon cross-asset reversal is hypothesized to work after a discrete
volatility shock; it is purged of the already-admitted vol-normalized reversal.
"""
import numpy as np,pandas as pd,glob,json
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-12-30'); P={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index();P[a]=pd.to_numeric(d.close,errors='coerce')
P=pd.DataFrame(P);R=P.pct_change(fill_method=None)
# candidate: raw 5d reversal, residualized daily cross-sectionally against
# canonical 5d vol-normalized reversal, retained only after a positive VIX shock.
raw=-(P/P.shift(5)-1); base=-((P/P.shift(5)-1)/R.rolling(5,min_periods=4).std())
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for dt in P.index:
 z=pd.concat([raw.loc[dt].rename('x'),base.loc[dt].rename('b')],axis=1).dropna()
 if len(z)>=8 and z.b.var()>0:
  beta=np.cov(z.x,z.b,ddof=1)[0,1]/z.b.var();F.loc[dt,z.index]=z.x-(z.x.mean()-beta*z.b.mean()+beta*z.b)
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index().close.astype(float).reindex(P.index)
shock=(v/v.shift(5)-1).clip(lower=0)
F=F.mul(shock,axis=0)
# reconstruct all library signals for mandatory orthogonality check
L={}
vol=R.rolling(20,min_periods=15).std();L['ravmom']=(P/P.shift(20)-1)/vol;L['volnorm_reversal']=base;L['realized_vol']=vol
peer=pd.DataFrame(index=P.index,columns=A)
for a in A: peer[a]=pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in A if b!=a],axis=1).mean(axis=1)
L['peer_crowding']=peer
def macro(fn):
 m=pd.read_csv('../persistent/index_data/'+fn,parse_dates=['date']).query('date<=@END').set_index('date').close.astype(float).pct_change(fill_method=None).reindex(P.index)
 x=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in A})
 for dt in x.index:
  z=pd.concat([x.loc[dt].rename('x'),peer.loc[dt].rename('p')],axis=1).dropna()
  if len(z)>=8 and z.p.var()>0:
   b=np.cov(z.x,z.p,ddof=1)[0,1]/z.p.var();x.loc[dt,z.index]=z.x-(z.x.mean()-b*z.p.mean()+b*z.p)
 return x
L['vix_beta_resid']=macro('VIX.csv');L['dxy_beta_resid']=macro('DXY.csv')
# reconstruct incumbent miner_3 signal as specified in its persisted definition
up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5);asym=np.log((up+1e-8)/(dn+1e-8));mom=P/P.shift(20)-1;res=pd.DataFrame(index=P.index,columns=A)
for dt in P.index:
 z=pd.concat([asym.loc[dt].rename('x'),mom.loc[dt].rename('m')],axis=1).dropna()
 if len(z)>=8 and z.m.var()>0:
  b=np.cov(z.x,z.m,ddof=1)[0,1]/z.m.var();res.loc[dt,z.index]=z.x-(z.x.mean()-b*z.m.mean()+b*z.m)
L['high_vix_asym']=res.mul((v/v.rolling(60,min_periods=40).mean()-1).clip(lower=0),axis=0)
def report(h):
 fw=P.shift(-h)/P-1; vals=[];nn=[]
 for dt in P.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1: vals.append((dt,z.f.corr(z.r,method='spearman')));nn.append(len(z))
 x=pd.Series(dict(vals)); ic=x.mean();ir=ic/x.std(ddof=1);print(f'h={h} dates={len(x)} IC={ic:.6f} ICIR={ir:.6f} hit={(x>0).mean():.4f} se={x.std(ddof=1)/np.sqrt(len(x)):.6f} coverage={np.mean(nn)/15:.4f}')
 if h==5:
  for n,q in [('2020_21',x.index<'2022-01-01'),('2022_23',(x.index>='2022-01-01')&(x.index<'2024-01-01')),('2024_25',(x.index>='2024-01-01')&(x.index<'2026-01-01')),('2026',x.index>='2026-01-01')]:
   y=x[q];print(n,len(y),f'IC={y.mean():.6f}',f'ICIR={y.mean()/y.std(ddof=1):.6f}')
print('FACTOR vix_shock_gated_residual_reversal_5; visible through',END.date(),'assets',len(A));print('expression=max(VIX/VIX[-5]-1,0)*residual_cs[-return5 ~ -return5/vol5]')
for h in [1,5,10,20]:report(h)
r=F.rank(axis=1,pct=True);tos=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:tos.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('turnover',np.mean(tos),'signal_cell_coverage',F.notna().mean().mean(),'shock_days',(shock>0).sum(),'/',shock.notna().sum())
mx=0
for n,x in L.items():
 z=pd.concat([F.stack().rename('f'),x.stack().rename('x')],axis=1).dropna();rho=z.f.corr(z.x,method='spearman');print('library',n,'rho',rho,'cells',len(z));mx=max(mx,abs(rho))
print('max_abs_library_correlation',mx,'records',len([x for x in glob.glob('factors/*.json') if not x.endswith('.bak')]))
