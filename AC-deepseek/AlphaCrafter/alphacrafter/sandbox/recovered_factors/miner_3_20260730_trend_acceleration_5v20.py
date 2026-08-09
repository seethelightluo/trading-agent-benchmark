"""miner_3 one-idea validation: 5-vs-20 observation trend acceleration, volatility-normalized."""
import glob, json
import numpy as np, pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
F={}; FW={}; L={}
# Reconstruct all admitted signal definitions from their JSON expressions.
for a in ASSETS:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
 p=pd.to_numeric(d.close,errors='coerce'); r=p.pct_change(fill_method=None)
 vol20=r.rolling(20,min_periods=15).std(); vol5=r.rolling(5,min_periods=4).std()
 # A positive value says recent 5-observation performance is stronger than the pace implied by the prior 20-observation trend.
 F[a]=(p.pct_change(5,fill_method=None)-p.pct_change(20,fill_method=None)/4)/vol20
 FW[a]={h:p.shift(-h)/p-1 for h in (1,5,10,20)}
 L.setdefault('miner_3_risk_adjusted_trend_20d',{})[a]=p.pct_change(20,fill_method=None)/vol20
 L.setdefault('miner_1_ravmom_20obs',{})[a]=p.pct_change(20,fill_method=None)/vol20
 L.setdefault('miner_1_volnorm_reversal_5obs',{})[a]=-p.pct_change(5,fill_method=None)/vol5
 if 'volume' in d:
  v=pd.to_numeric(d.volume,errors='coerce'); L.setdefault('miner_3_relative_volume_participation_20d',{})[a]=np.log(v/v.rolling(20,min_periods=15).mean())
 L.setdefault('miner_2_realized_volatility_20obs',{})[a]=-vol20
f=pd.DataFrame(F).sort_index()
print('FACTOR trend_acceleration_5v20_volnorm = (return_5 - return_20/4) / rolling_std(return,20); higher = recent trend acceleration beyond its 20-observation pace')
print('history',f.index.min().date(),f.index.max().date(),'instruments',len(ASSETS))
def calc(h):
 fw=pd.DataFrame({a:FW[a][h] for a in ASSETS}); out=[]; cov=[]
 for dt in f.index:
  z=pd.DataFrame({'factor':f.loc[dt],'forward':fw.loc[dt]}).dropna()
  if len(z)>=8: out.append((dt,z.factor.corr(z.forward,method='spearman'))); cov.append(len(z)/15)
 x=pd.Series(dict(out)); return x,float(np.mean(cov))
allres={}
for h in (1,5,10,20):
 x,c=calc(h); allres[h]=x; sd=x.std(ddof=1)
 print(f'h={h} dates={len(x)} meanIC={x.mean():.6f} ICIR={x.mean()/sd:.6f} hit={(x>0).mean():.4f} IC_se={sd/np.sqrt(len(x)):.6f} coverage={c:.4f}')
 for nm,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
  z=x[mask]; print(f'  {nm}: n={len(z)} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f}')
r=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(r)):
 z=pd.concat([r.iloc[i-1],r.iloc[i]],axis=1).dropna()
 if len(z)>=8: turns.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover_rank_change={np.mean(turns):.6f}; valid_signal_cells={f.notna().sum().sum()}/{f.size} ({f.notna().mean().mean():.4f})')
mx=0; evidence={}
for name,vals in L.items():
 z=pd.concat([f.stack().rename('candidate'),pd.DataFrame(vals).stack().rename('library')],axis=1).dropna()
 rho=z.candidate.corr(z.library,method='spearman'); evidence[name]=(rho,len(z)); mx=max(mx,abs(rho)); print(f'library {name}: rho={rho:.6f} cells={len(z)}')
print(f'LIBRARY records={len([x for x in glob.glob("factors/*.json") if not x.endswith(".bak")])} max_abs_library_correlation={mx:.6f}')
