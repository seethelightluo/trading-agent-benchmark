"""miner_1 validation: volatility-normalized cross-sectional relative strength."""
import glob,pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index()
R=P.pct_change(fill_method=None); V=R.rolling(20,min_periods=15).std(); mom=P.pct_change(20,fill_method=None)
# Cross-sectional relative return divided by each asset's realized risk; robustly removes common cross-asset drift while favoring efficient winners.
F=mom.sub(mom.median(axis=1),axis=0).div(V)
FW={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
print('FACTOR volnorm_relative_strength_20d = (return_20 - cross-sectional_median(return_20))/rolling_std(return,20)')
def calc(h):
 out=[]; cov=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':FW[h].loc[dt]}).dropna()
  if len(z)>=8: out.append((dt,z.f.corr(z.y,method='spearman'))); cov.append(len(z)/15)
 x=pd.Series(dict(out)); return x,np.mean(cov)
for h in (1,5,10,20):
 x,c=calc(h); sd=x.std(ddof=1); print(f'h={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} IC_se={sd/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 for nm,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
  z=x[mask]; print(f'  {nm}: n={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f}')
rr=F.rank(axis=1,pct=True); ts=[]
for i in range(1,len(rr)):
 z=pd.concat([rr.iloc[i-1],rr.iloc[i]],axis=1).dropna()
 if len(z)>=8: ts.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover_rank_change={np.mean(ts):.6f}; valid_signal_cells={F.notna().sum().sum()}/{F.size} ({F.notna().mean().mean():.4f})')
# admitted signal reconstructions
L={}
for a in A:
 p=P[a]; r=p.pct_change(fill_method=None); v5=r.rolling(5,min_periods=4).std(); v20=r.rolling(20,min_periods=15).std()
 L['ravmom_20obs',a]=p.pct_change(20,fill_method=None)/v20; L['volnorm_reversal_5obs',a]=-p.pct_change(5,fill_method=None)/v5; L['realized_volatility_20obs',a]=-v20
 if a in A:
  d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date'); v=pd.to_numeric(d.volume,errors='coerce'); L['relative_volume',a]=np.log(v/v.rolling(20,min_periods=15).mean())
mx=0
for n in sorted(set(k[0] for k in L)):
 q=pd.DataFrame({a:L[n,a] for a in A}); z=pd.concat([F.stack().rename('c'),q.stack().rename('l')],axis=1).dropna(); rho=z.c.corr(z.l,method='spearman'); mx=max(mx,abs(rho)); print(f'library {n}: rho={rho:.6f} cells={len(z)}')
print(f'LIBRARY records=5 max_abs_library_correlation={mx:.6f}')
