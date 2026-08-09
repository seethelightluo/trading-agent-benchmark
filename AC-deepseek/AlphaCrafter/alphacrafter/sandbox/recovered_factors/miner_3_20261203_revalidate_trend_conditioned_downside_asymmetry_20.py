"""miner_3 revalidation: trend-conditioned downside asymmetry, one existing idea."""
import numpy as np, pandas as pd, glob
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-12-02'); HS=(1,5,10,20); ret={}; px={}; F={}
for a in ASSETS:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).query('date<=@END').drop_duplicates('date').set_index('date').sort_index()
 p=pd.to_numeric(d.close,errors='coerce'); r=p.pct_change(fill_method=None); px[a]=p; ret[a]=r
 up=r.where(r>0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5); dn=r.where(r<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
 F[a]=np.sign(p/p.shift(20)-1)*np.log((up+1e-8)/(dn+1e-8))
f=pd.DataFrame(F).sort_index(); R=pd.DataFrame(ret).sort_index()
L={k:{} for k in ['miner_1_ravmom_20obs','miner_1_volnorm_reversal_5obs','miner_2_realized_volatility_20obs','miner_2_peer_crowding_correlation_20obs','miner_1_vix_beta_residual_peer20','miner_1_dxy_beta_residual_peer20']}
for a in ASSETS:
 p=px[a];r=R[a];v=r.rolling(20,min_periods=15).std();L['miner_1_ravmom_20obs'][a]=(p/p.shift(20)-1)/v;L['miner_1_volnorm_reversal_5obs'][a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std();L['miner_2_realized_volatility_20obs'][a]=v
for a in ASSETS:L['miner_2_peer_crowding_correlation_20obs'][a]=pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in ASSETS if b!=a],axis=1).mean(axis=1)
def macro_resid(file,key):
 m=pd.read_csv('../persistent/index_data/'+file,parse_dates=['date']).query('date<=@END').set_index('date').close.astype(float).pct_change(fill_method=None).reindex(R.index)
 raw=pd.DataFrame({a:R[a].rolling(20,min_periods=15).corr(m) for a in ASSETS});peer=pd.DataFrame(L['miner_2_peer_crowding_correlation_20obs'])
 for dt in raw.index:
  z=pd.concat([raw.loc[dt],peer.loc[dt]],axis=1).dropna()
  if len(z)>=2:
   b=np.cov(z.iloc[:,0],z.iloc[:,1],ddof=1)[0,1]/np.var(z.iloc[:,1],ddof=1);raw.loc[dt]=raw.loc[dt]-(z.iloc[:,0].mean()-b*z.iloc[:,1].mean()+b*peer.loc[dt])
 for a in ASSETS:L[key][a]=raw[a]
macro_resid('VIX.csv','miner_1_vix_beta_residual_peer20');macro_resid('DXY.csv','miner_1_dxy_beta_residual_peer20')
def stats(x):
 return x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),x.std(ddof=1)/np.sqrt(len(x))
def getic(h):
 fw=pd.DataFrame({a:px[a].shift(-h)/px[a]-1 for a in ASSETS});o=[];cv=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('s'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8:o.append((dt,z.s.corr(z.r,method='spearman')));cv.append(len(z)/15)
 return pd.Series(dict(o)),np.mean(cv)
print('FACTOR trend_conditioned_downside_asymmetry_20; visible through',END.date(),'assets',len(ASSETS),'history',f.index.min().date(),f.index.max().date())
for h in HS:
 x,c=getic(h);m,ir,hit,se=stats(x);print(f'h={h} dates={len(x)} IC={m:.6f} ICIR={ir:.6f} hit={hit:.4f} se={se:.6f} coverage={c:.4f} mean_n={15*c:.2f}')
 if h in (5,10):
  for n,mask in [('2020',x.index<'2021'),('2021_22',(x.index>='2021')&(x.index<'2023')),('2023_24',(x.index>='2023')&(x.index<'2025')),('2025', (x.index>='2025')&(x.index<'2026')),('2026',x.index>='2026')]:
   q=x[mask]; a,b,d,e=stats(q);print(f'  h={h} {n} dates={len(q)} IC={a:.6f} ICIR={b:.6f} hit={d:.4f}')
r=f.rank(axis=1,pct=True); t=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8:t.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(t):.6f} signal_coverage={f.notna().mean().mean():.4f}')
mx=0
for n,v in L.items():
 z=pd.concat([f.stack().rename('x'),pd.DataFrame(v).stack().rename('y')],axis=1).dropna();rho=z.x.corr(z.y,method='spearman');mx=max(mx,abs(rho));print(f'library {n} rho={rho:.6f} cells={len(z)}')
print('library_records',len(glob.glob('factors/*.json')),'max_abs_library_correlation',f'{mx:.6f}')
