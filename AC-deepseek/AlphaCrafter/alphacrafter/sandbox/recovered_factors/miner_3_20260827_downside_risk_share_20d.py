"""miner_3 one-idea validation: downside-risk share (20 daily observations)."""
import glob
import numpy as np
import pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=(1,5,10,20)
F={}; FW={}; LIB={}
for a in ASSETS:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
 p=pd.to_numeric(d.close,errors='coerce'); r=p.pct_change(fill_method=None); vol=r.rolling(20,min_periods=15).std()
 # Fraction of total variation arising from returns below zero. Lower values mean more favorable/upside-dominated risk.
 downside=r.where(r<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)
 F[a]=-downside/vol
 FW[a]={h:p.shift(-h)/p-1 for h in H}
 LIB.setdefault('miner_3_risk_adjusted_trend_20d',{})[a]=(p/p.shift(20)-1)/vol
 LIB.setdefault('miner_3_relative_volume_participation_20d',{})[a]=np.log(pd.to_numeric(d.volume,errors='coerce')/pd.to_numeric(d.volume,errors='coerce').rolling(20,min_periods=15).mean())
 LIB.setdefault('miner_1_ravmom_20obs',{})[a]=(p/p.shift(20)-1)/vol
 LIB.setdefault('miner_2_realized_volatility_20obs',{})[a]=vol
 LIB.setdefault('miner_1_volnorm_reversal_5obs',{})[a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
 # Mean 20-day correlation with the other 14 returns, as used by admitted peer-crowding factor.
 LIB.setdefault('miner_2_peer_crowding_correlation_20obs',{})[a]=r.rolling(20,min_periods=15).corr(r) # placeholder replaced below
 # calculate per-asset peer mean jointly after raw returns are retained
 if 'RET' not in globals(): RET={}
 RET[a]=r
R=pd.DataFrame(RET).sort_index()
for a in ASSETS:
 peers=[x for x in ASSETS if x!=a]
 LIB['miner_2_peer_crowding_correlation_20obs'][a]=pd.concat([R[a].rolling(20,min_periods=15).corr(R[b]) for b in peers],axis=1).mean(axis=1)
f=pd.DataFrame(F).sort_index(); print('FACTOR: downside_risk_share_20d = -sqrt(mean(min(return,0)^2,20))/stdev(return,20)'); print('history',f.index.min().date(),f.index.max().date(),'instruments',len(ASSETS))
def st(x):
 sd=x.std(ddof=1); return x.mean(),x.mean()/sd,(x>0).mean(),sd/np.sqrt(len(x))
def calc(h):
 fw=pd.DataFrame({a:FW[a][h] for a in ASSETS}); obs=[]; cov=[]
 for dt in f.index:
  z=pd.DataFrame({'s':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(z)>=8: obs.append((dt,z.s.corr(z.r,method='spearman'))); cov.append(len(z)/15)
 return pd.Series(dict(obs)),np.mean(cov)
for h in H:
 x,cov=calc(h); m,ir,hit,se=st(x); print(f'h={h} dates={len(x)} meanIC={m:.6f} ICIR={ir:.6f} hit={hit:.4f} IC_se={se:.6f} coverage={cov:.4f}')
 for nm,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
  q=x[mask]; m1,i1,h1,_=st(q); print(f'  {nm}: n={len(q)} IC={m1:.6f} ICIR={i1:.6f} hit={h1:.4f}')
rnk=f.rank(axis=1,pct=True); changes=[]
for i in range(1,len(rnk)):
 z=pd.concat([rnk.iloc[i-1],rnk.iloc[i]],axis=1).dropna()
 if len(z)>=8: changes.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover_rank_change={np.mean(changes):.6f}; valid_signal_cells={f.notna().sum().sum()}/{f.size} ({f.notna().mean().mean():.4f})')
mx=0
for n,v in LIB.items():
 z=pd.concat([f.stack().rename('candidate'),pd.DataFrame(v).stack().rename('library')],axis=1).dropna(); rho=z.candidate.corr(z.library,method='spearman'); mx=max(mx,abs(rho)); print(f'library {n}: rho={rho:.6f} cells={len(z)}')
print(f'LIBRARY records={len(glob.glob("factors/*.json"))} max_abs_library_correlation={mx:.6f}')
