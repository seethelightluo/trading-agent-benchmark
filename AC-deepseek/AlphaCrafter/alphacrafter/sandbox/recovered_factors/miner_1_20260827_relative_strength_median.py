"""miner_1 validation: 20-day relative strength versus cross-sectional median."""
import glob,json
import numpy as np,pandas as pd
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in A}).sort_index()
R=P.pct_change(fill_method=None)
# Relative 20d return subtracts the contemporaneous cross-sectional median, reducing common beta.
raw=P.pct_change(20,fill_method=None)
F=raw.sub(raw.median(axis=1),axis=0)
FW={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
print('FACTOR relative_strength_median_20d = return_20 - cross_sectional_median(return_20); higher predicts higher forward return')
print('history',F.index.min().date(),F.index.max().date(),'instruments',len(A))
def calc(h):
 fw=FW[h]; out=[]; cov=[]
 for dt in F.index:
  z=pd.DataFrame({'f':F.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8: out.append((dt,z.f.corr(z.y,method='spearman'))); cov.append(len(z)/len(A))
 x=pd.Series(dict(out)); return x,np.mean(cov)
for h in (1,5,10,20):
 x,c=calc(h); sd=x.std(ddof=1)
 print(f'h={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} IC_se={sd/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 for nm,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
  z=x[mask]; print(f'  {nm}: n={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
rr=F.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rr)):
 z=pd.concat([rr.iloc[i-1],rr.iloc[i]],axis=1).dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover_rank_change={np.mean(turns):.6f}; valid_signal_cells={F.notna().sum().sum()}/{F.size} ({F.notna().mean().mean():.4f})')
# reconstruct admitted library signals for pooled Spearman evidence
L={}
for a in A:
 p=P[a]; r=p.pct_change(fill_method=None); v5=r.rolling(5,min_periods=4).std(); v20=r.rolling(20,min_periods=15).std()
 L.setdefault('miner_1_ravmom_20obs',{})[a]=p.pct_change(20,fill_method=None)/v20
 L.setdefault('miner_1_volnorm_reversal_5obs',{})[a]=-p.pct_change(5,fill_method=None)/v5
 L.setdefault('miner_2_realized_volatility_20obs',{})[a]=-v20
 L.setdefault('miner_3_risk_adjusted_trend_20d',{})[a]=p.pct_change(20,fill_method=None)/v20
 if a in A:
  d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')
  if 'volume' in d: L.setdefault('miner_3_relative_volume_participation_20d',{})[a]=np.log(pd.to_numeric(d.volume,errors='coerce')/pd.to_numeric(d.volume,errors='coerce').rolling(20,min_periods=15).mean())
mx=0
for n,v in L.items():
 z=pd.concat([F.stack().rename('candidate'),pd.DataFrame(v).stack().rename('lib')],axis=1).dropna(); rho=z.candidate.corr(z.lib,method='spearman'); mx=max(mx,abs(rho)); print(f'library {n}: rho={rho:.6f} cells={len(z)}')
print(f'LIBRARY records={len([x for x in glob.glob("factors/*.json") if not x.endswith(".bak")])} max_abs_library_correlation={mx:.6f}')
