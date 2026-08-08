"""miner_3 revalidation: one factor, 20-observation relative-volume participation."""
import pandas as pd, numpy as np, json, glob
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-10-21'); H=[1,5,10,20]
P={}; R={}; V={}
for a in ASSETS:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index().loc[:END]
 P[a]=d.close.astype(float); R[a]=P[a].pct_change(); V[a]=d.volume.astype(float).replace(0,np.nan)
p=pd.DataFrame(P); r=pd.DataFrame(R); v=pd.DataFrame(V)
f=np.log(v/v.rolling(20,min_periods=15).mean())
# Reconstruct every currently admitted library signal from stated definitions.
lib={
 'miner_1_ravmom_20obs':(p/p.shift(20)-1)/r.rolling(20,min_periods=15).std(),
 'miner_1_volnorm_reversal_5obs':-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std(),
 'miner_2_realized_volatility_20obs':r.rolling(20,min_periods=15).std(),
}
peer={}
for a in ASSETS:
 peer[a]=r.drop(columns=a).mean(axis=1)
lib['miner_2_peer_crowding_correlation_20obs']=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(peer[a]) for a in ASSETS})
print('FACTOR log(volume_t / rolling_mean(volume,20)); visible through',END.date())
print('history',f.index.min().date(),f.index.max().date(),'assets',len(ASSETS),'signal cells',f.notna().sum().sum(),'/',f.size)
def getic(h):
 fw=p.shift(-h)/p-1; arr=[]; cov=[]; nvalid=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt].rename('x'),fw.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: arr.append((dt,z.x.corr(z.y,method='spearman'))); cov.append(len(z)/15); nvalid.append(len(z))
 return pd.Series(dict(arr)),np.mean(cov),np.mean(nvalid)
for h in H:
 x,c,n=getic(h); sd=x.std(ddof=1)
 print(f'H{h} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} se={sd/np.sqrt(len(x)):.6f} coverage={c:.4f} meanN={n:.2f}')
 if h==10:
  for name,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
   y=x[mask]; print(f' REGIME {name} n={len(y)} IC={y.mean():.6f} ICIR={y.mean()/y.std(ddof=1):.6f} hit={(y>0).mean():.4f}')
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8: turn.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print('mean_abs_rank_change',np.mean(turn))
mx=0; evidence={}
for name,x in lib.items():
 z=pd.concat([f.stack().rename('factor'),x.stack().rename('library')],axis=1).dropna(); rho=z.factor.corr(z.library,method='spearman'); evidence[name]=(rho,len(z)); mx=max(mx,abs(rho)); print(f'LIB {name} rho={rho:.6f} cells={len(z)}')
print('MAX_ABS_LIBRARY_CORR',mx,'library_records',len([x for x in glob.glob('factors/*.json') if not x.endswith('_deprecated.json')]))
