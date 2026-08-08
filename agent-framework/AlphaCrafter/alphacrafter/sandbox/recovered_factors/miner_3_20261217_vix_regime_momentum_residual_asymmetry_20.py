"""miner_3: one idea -- VIX-regime-conditioned momentum-residualized downside asymmetry."""
import numpy as np, pandas as pd, glob
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-12-16'); HS=(1,5,10,20); px={}; rr={}
for a in ASSETS:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index()
 px[a]=pd.to_numeric(d.close,errors='coerce'); rr[a]=px[a].pct_change(fill_method=None)
R=pd.DataFrame(rr).sort_index(); P=pd.DataFrame(px).sort_index()
# Path-quality log asymmetry, purged each day of cross-sectional 20d trend exposure.
up=R.where(R>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
dn=R.where(R<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
asym=np.log((up+1e-8)/(dn+1e-8)); mom=P/P.shift(20)-1
res=pd.DataFrame(index=R.index,columns=ASSETS,dtype=float)
for dt in R.index:
 z=pd.concat([asym.loc[dt].rename('a'),mom.loc[dt].rename('m')],axis=1).dropna()
 if len(z)>=8:
  beta=np.cov(z.a,z.m,ddof=1)[0,1]/np.var(z.m,ddof=1)
  res.loc[dt,z.index]=z.a-(z.a.mean()-beta*z.m.mean()+beta*z.m)
# VIX high-volatility state is observable macro context.  Only in stressed state
# is residual path asymmetry retained; quiet-state factor is zero/uninformative.
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index().close.astype(float).reindex(R.index)
state=(v/v.rolling(60,min_periods=40).mean()-1).clip(lower=0)
F=res.mul(state,axis=0)
# admitted signals reconstructed exactly for correlation evidence
L={k:{} for k in ['miner_1_ravmom_20obs','miner_1_volnorm_reversal_5obs','miner_2_realized_volatility_20obs','miner_2_peer_crowding_correlation_20obs','miner_1_vix_beta_residual_peer20','miner_1_dxy_beta_residual_peer20']}
for a in ASSETS:
 r=R[a];p=P[a];vol=r.rolling(20,min_periods=15).std();L['miner_1_ravmom_20obs'][a]=(p/p.shift(20)-1)/vol;L['miner_1_volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();L['miner_2_realized_volatility_20obs'][a]=vol
for a in ASSETS:L['miner_2_peer_crowding_correlation_20obs'][a]=pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in ASSETS if b!=a],axis=1).mean(axis=1)
def macro(file,key):
 m=pd.read_csv('../persistent/index_data/'+file,parse_dates=['date']).query('date<=@END').set_index('date').close.astype(float).pct_change(fill_method=None).reindex(R.index);raw=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in ASSETS});peer=pd.DataFrame(L['miner_2_peer_crowding_correlation_20obs'])
 for dt in raw.index:
  z=pd.concat([raw.loc[dt].rename('x'),peer.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=2:
   b=np.cov(z.x,z.y,ddof=1)[0,1]/np.var(z.y,ddof=1);raw.loc[dt]=raw.loc[dt]-(z.x.mean()-b*z.y.mean()+b*peer.loc[dt])
 for a in ASSETS:L[key][a]=raw[a]
macro('VIX.csv','miner_1_vix_beta_residual_peer20');macro('DXY.csv','miner_1_dxy_beta_residual_peer20')
def stat(x):
 sd=x.std(ddof=1);return x.mean(),x.mean()/sd,(x>0).mean(),sd/np.sqrt(len(x))
def getic(h):
 fw=P.shift(-h)/P-1;out=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  # State exactly zero has no cross-sectional ranking and cannot define an IC.
  if len(z)>=8 and z.f.nunique()>1:out.append((dt,z.f.corr(z.r,method='spearman')));ns.append(len(z))
 return pd.Series(dict(out)),np.mean(ns)/15
print('FACTOR: high_vix_momentum_residual_downside_asymmetry_20; visible through',END.date(),'instruments=15')
print('expression: max(VIX/mean(VIX,60)-1,0) * residual_cs[log(up_semidev20/down_semidev20) ~ return20]')
for h in HS:
 x,c=getic(h);a,b,d,se=stat(x);print(f'h={h} dates={len(x)} IC={a:.6f} ICIR={b:.6f} hit={d:.4f} se={se:.6f} coverage={c:.4f} mean_n={15*c:.2f}')
 if h==5:
  for n,q in [('2020_21',x.index<'2022-01-01'),('2022_23',(x.index>='2022-01-01')&(x.index<'2024-01-01')),('2024_25',(x.index>='2024-01-01')&(x.index<'2026-01-01')),('2026',x.index>='2026-01-01')]:
   y=x[q];aa,bb,dd,_=stat(y);print(f' {n} dates={len(y)} IC={aa:.6f} ICIR={bb:.6f} hit={dd:.4f}')
r=F.rank(axis=1,pct=True);to=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:to.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(to):.6f} signal_cell_coverage={F.notna().mean().mean():.4f} nonzero_state_days={(state>0).sum()}/{state.notna().sum()}')
mx=0
for n,vv in L.items():
 z=pd.concat([F.stack().rename('x'),pd.DataFrame(vv).stack().rename('y')],axis=1).dropna();rho=z.x.corr(z.y,method='spearman');mx=max(mx,abs(rho));print(f'library {n} rho={rho:.6f} cells={len(z)}')
print('max_abs_library_correlation',f'{mx:.6f}','library_json_records',len(glob.glob('factors/*.json')))
