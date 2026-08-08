"""miner_3 one-idea validation: cross-sectionally residualized 20d downside-risk share."""
import glob
import numpy as np
import pandas as pd
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; H=(1,5,10,20)
RAW={}; TREND={}; VOL={}; RVOL={}; REV={}; RET={}; CLOSE={}
for a in ASSETS:
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).set_index('date').sort_index()
 p=pd.to_numeric(d.close,errors='coerce'); r=p.pct_change(fill_method=None); v=r.rolling(20,min_periods=15).std(); volume=pd.to_numeric(d.volume,errors='coerce')
 CLOSE[a]=p; RET[a]=r; VOL[a]=v
 RAW[a]=-r.where(r<0,0).pow(2).rolling(20,min_periods=15).mean().pow(.5)/v
 TREND[a]=(p/p.shift(20)-1)/v; RVOL[a]=np.log(volume/volume.rolling(20,min_periods=15).mean()); REV[a]=-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std()
raw=pd.DataFrame(RAW).sort_index(); trend=pd.DataFrame(TREND).reindex(raw.index); vol=pd.DataFrame(VOL).reindex(raw.index); rvol=pd.DataFrame(RVOL).reindex(raw.index); rev=pd.DataFrame(REV).reindex(raw.index)
r=pd.DataFrame(RET).sort_index(); peer=pd.DataFrame(index=raw.index,columns=ASSETS,dtype=float)
for a in ASSETS: peer[a]=pd.concat([r[a].rolling(20,min_periods=15).corr(r[b]) for b in ASSETS if b!=a],axis=1).mean(axis=1)
# Each date: residual of downside-share rank after intercept + risk-adjusted-trend rank.
# This retains cross-asset tail asymmetry not explained by concurrent trend.
f=pd.DataFrame(index=raw.index,columns=ASSETS,dtype=float)
for dt in raw.index:
 y=raw.loc[dt].rank(pct=True); x=trend.loc[dt].rank(pct=True); ok=y.notna()&x.notna()
 if ok.sum()>=8:
  X=np.column_stack([np.ones(ok.sum()),x[ok].values]); beta=np.linalg.lstsq(X,y[ok].values,rcond=None)[0]
  f.loc[dt,ok]=y[ok]-X@beta
print('FACTOR: residualized_downside_risk_share_20d = daily cross-sectional rank(downside-share) residual after rank(risk-adjusted 20d trend)')
print('history',f.index.min().date(),f.index.max().date(),'instruments',len(ASSETS))
def stats(x):
 sd=x.std(ddof=1); return x.mean(),x.mean()/sd,(x>0).mean(),sd/np.sqrt(len(x))
def ics(h):
 fw=pd.DataFrame({a:CLOSE[a].shift(-h)/CLOSE[a]-1 for a in ASSETS}).reindex(f.index); out=[]; coverage=[]
 for dt in f.index:
  z=pd.DataFrame({'s':f.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(z)>=8: out.append((dt,z.s.corr(z.r,method='spearman'))); coverage.append(len(z)/15)
 return pd.Series(dict(out)),np.mean(coverage)
for h in H:
 x,c=ics(h); m,ir,hit,se=stats(x); print(f'h={h} dates={len(x)} meanIC={m:.6f} ICIR={ir:.6f} hit={hit:.4f} IC_se={se:.6f} coverage={c:.4f}')
 if h==20:
  for n,mask in [('2020',x.index<'2021-01-01'),('2021_22',(x.index>='2021-01-01')&(x.index<'2023-01-01')),('2023_24',(x.index>='2023-01-01')&(x.index<'2025-01-01')),('2025_26',x.index>='2025-01-01')]:
   q=x[mask]; a,b,c1,_=stats(q); print(f'  {n}: n={len(q)} IC={a:.6f} ICIR={b:.6f} hit={c1:.4f}')
ranks=f.rank(axis=1,pct=True); changes=[]
for i in range(1,len(ranks)):
 z=pd.concat([ranks.iloc[i-1],ranks.iloc[i]],axis=1).dropna()
 if len(z)>=8: changes.append(1-z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
print(f'turnover_rank_change={np.mean(changes):.6f}; valid_signal_cells={f.notna().sum().sum()}/{f.size} ({f.notna().mean().mean():.4f})')
libs={'miner_3_risk_adjusted_trend_20d':trend,'miner_1_ravmom_20obs':trend,'miner_3_relative_volume_participation_20d':rvol,'miner_1_volnorm_reversal_5obs':rev,'miner_2_realized_volatility_20obs':vol,'miner_2_peer_crowding_correlation_20obs':peer}
mx=0
for name,L in libs.items():
 z=pd.concat([f.stack().rename('f'),L.stack().rename('l')],axis=1).dropna(); rho=z.f.corr(z.l,method='spearman'); mx=max(mx,abs(rho)); print(f'library {name}: rho={rho:.6f} cells={len(z)}')
print(f'LIBRARY records={len(glob.glob("factors/*.json"))} max_abs_library_correlation={mx:.6f}')
