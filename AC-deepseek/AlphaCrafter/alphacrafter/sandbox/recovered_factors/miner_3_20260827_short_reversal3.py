"""Validate one idea: short-horizon 3-observation reversal, volatility normalized."""
import glob, numpy as np, pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}; FW={}; L={}
for a in ASSETS:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
 p=pd.to_numeric(d.close,errors='coerce'); r=p.pct_change(fill_method=None)
 v10=r.rolling(10,min_periods=8).std()
 F[a]=-p.pct_change(3,fill_method=None)/v10
 FW[a]={h:p.shift(-h)/p-1 for h in (1,5,10,20)}
 L.setdefault('miner_3_risk_adjusted_trend_20d',{})[a]=p.pct_change(20,fill_method=None)/r.rolling(20,min_periods=15).std()
 L.setdefault('miner_1_volnorm_reversal_5obs',{})[a]=-p.pct_change(5,fill_method=None)/r.rolling(5,min_periods=4).std()
 L.setdefault('miner_3_relative_volume_participation_20d',{})[a]=np.log(pd.to_numeric(d.volume,errors='coerce')/pd.to_numeric(d.volume,errors='coerce').rolling(20,min_periods=15).mean()) if 'volume' in d else pd.Series(index=p.index)
 L.setdefault('miner_2_realized_volatility_20obs',{})[a]=-r.rolling(20,min_periods=15).std()
f=pd.DataFrame(F).sort_index(); print('FACTOR short_reversal_3obs_volnorm = -return_3 / rolling_std(return,10)'); print('history',f.index.min().date(),f.index.max().date(),'instruments',len(ASSETS))
def calc(h):
 fw=pd.DataFrame({a:FW[a][h] for a in ASSETS}); out=[]; cov=[]
 for dt in f.index:
  z=pd.DataFrame({'x':f.loc[dt],'y':fw.loc[dt]}).dropna()
  if len(z)>=8: out.append((dt,z.x.corr(z.y,method='spearman'))); cov.append(len(z)/15)
 x=pd.Series(dict(out)); return x,np.mean(cov)
for h in (1,5,10,20):
 x,c=calc(h); print(f'h={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1):.6f} hit={(x>0).mean():.4f} coverage={c:.4f}')
 for nm,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
  z=x[mask]; print(f' {nm}: n={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
r=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover={np.mean(turns):.6f} coverage_cells={f.notna().mean().mean():.4f}')
mx=0
for name,vals in L.items():
 z=pd.concat([f.stack().rename('c'),pd.DataFrame(vals).stack().rename('l')],axis=1).dropna(); rho=z.c.corr(z.l,method='spearman'); mx=max(mx,abs(rho)); print(f'library {name}: rho={rho:.6f} cells={len(z)}')
print(f'max_abs_library_correlation={mx:.6f}')
