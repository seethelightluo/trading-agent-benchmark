"""miner_3: test 20-observation serial-return persistence, visible through 2026-11-04."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-11-04'); idx=pd.date_range('2020-01-01',END,freq='B')
raw={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 raw[a]=d.close.replace(0,np.nan).astype(float)
px=pd.DataFrame(raw).reindex(idx).ffill(); r=px.pct_change()
# Candidate: sample lag-1 autocorrelation of each asset's last 20 daily returns.
def ac1(x):
 x=np.asarray(x); ok=np.isfinite(x)
 x=x[ok]
 if len(x)<15:return np.nan
 if np.std(x[:-1])==0 or np.std(x[1:])==0:return np.nan
 return np.corrcoef(x[:-1],x[1:])[0,1]
cand=r.rolling(20,min_periods=15).apply(ac1,raw=True)
rv=r.rolling(20,min_periods=15).std()
trend=(px/px.shift(20)-1)/rv
rev=-(px/px.shift(5)-1)/r.rolling(5,min_periods=4).std()
def peer(x):
 out=pd.DataFrame(index=x.index,columns=x.columns,dtype=float)
 for a in x: out[a]=x[a].rolling(20,min_periods=15).corr(x.drop(columns=a).mean(axis=1))
 return out
pc=peer(r)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.reindex(idx).ffill().pct_change()
vb=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(vix) for a in A})
# exact cross-sectional residual of VIX beta upon peer crowding
vbr=pd.DataFrame(index=idx,columns=A,dtype=float)
for dt in idx:
 z=pd.concat([vb.loc[dt],pc.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  X=np.c_[np.ones(len(z)),z.iloc[:,1]]; beta=np.linalg.lstsq(X,z.iloc[:,0],rcond=None)[0]
  vbr.loc[dt,z.index]=z.iloc[:,0]-X@beta
lib={'ravmom_20obs':trend,'volnorm_reversal_5obs':rev,'realized_volatility_20obs':rv,'peer_crowding_correlation_20obs':pc,'vix_beta_residual_peer20':vbr}
print('FACTOR serial_return_persistence_20 = corr(r[t-19:t-1], r[t-18:t]); positive signals short-horizon continuation')
print('visible_through',END.date(),'grid_dates',len(idx),'assets',len(A),'signal_cells',int(cand.notna().sum().sum()),'coverage',round(cand.notna().mean().mean(),6))
ics={}
for h in [1,5,10,20]:
 fw=px.shift(-h)/px-1; vals=[]; nn=[]
 for dt in idx:
  z=pd.concat([cand.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic));nn.append(len(z))
 s=pd.Series(dict(vals)); ics[h]=s; sd=s.std(ddof=1)
 print(f'h={h} dates={len(s)} IC={s.mean():.6f} ICIR={s.mean()/sd:.6f} hit={(s>0).mean():.4f} se={sd/np.sqrt(len(s)):.6f} mean_n={np.mean(nn):.2f}')
for name,years in [('2020',[2020]),('2021_22',[2021,2022]),('2023_24',[2023,2024]),('2025_26',[2025,2026])]:
 s=ics[10][ics[10].index.year.isin(years)];print(f'regime_10d {name} dates={len(s)} IC={s.mean():.6f} ICIR={s.mean()/s.std(ddof=1):.6f} hit={(s>0).mean():.4f}')
rk=cand.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 z=pd.concat([rk.iloc[i-1],rk.iloc[i]],axis=1).dropna()
 if len(z)>=8: turns.append(1-spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('mean_one_day_rank_turnover',round(float(np.mean(turns)),6))
for n,x in lib.items():
 z=pd.concat([cand.stack(),x.stack()],axis=1).dropna(); rho=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
 print(f'library_corr {n} rho={rho:.6f} common_cells={len(z)}')
cand.to_pickle('scripts/miner_3_20261105_serial_return_persistence_20_signal.pkl')
