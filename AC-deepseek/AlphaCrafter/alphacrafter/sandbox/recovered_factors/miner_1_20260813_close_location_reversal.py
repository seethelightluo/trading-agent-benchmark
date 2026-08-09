"""Validate one candidate: 5-observation close-location reversal, using native asset calendars."""
import os, glob
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2026-08-12')
base='../persistent/stock_data'
# candidate: negative mean CLV (close location in own daily range), a weak-close mean-reversion proxy
signals={}; closes={}
for a in ASSETS:
    f=os.path.join(base,a+'.csv'); d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); d=d[d.date<=END].copy()
    rng=d.high-d.low
    clv=((d.close-d.low)/rng).where(rng>0) # 0 low, 1 high
    signals[a]=pd.Series(-clv.rolling(5,min_periods=4).mean().to_numpy(),index=d.date)
    closes[a]=pd.Series(d.close.to_numpy(),index=d.date)
# each fwd return uses own native observations, never an outer calendar shift
rows=[]
for a in ASSETS:
    for h in [1,5,10,20]:
        x=pd.DataFrame({'signal':signals[a], 'forward':closes[a].shift(-h)/closes[a]-1})
        x['asset']=a; x['h']=h; x['date']=x.index; rows.append(x.reset_index(drop=True))
panel=pd.concat(rows,ignore_index=True).dropna()

def ic_metrics(h):
    z=panel[panel.h==h]; ics=[]; ns=[]
    for dt,g in z.groupby('date'):
        if len(g)>=8:
            r=spearmanr(g.signal,g.forward).statistic
            if np.isfinite(r): ics.append((dt,r)); ns.append(len(g))
    q=pd.DataFrame(ics,columns=['date','ic']); mean=q.ic.mean(); std=q.ic.std(ddof=1)
    return q,mean,mean/std,(q.ic>0).mean(),len(q),np.mean(ns)
print('CANDIDATE: -rolling_mean_5((close-low)/(high-low)); through',END.date())
print('signal coverage cells',len(panel[panel.h==1])/(len(ASSETS)*len(pd.date_range('2020-01-01',END,freq='D'))))
for h in [1,5,10,20]:
 q,m,ir,hit,n,nv=ic_metrics(h); print(f'H{h}: IC={m:.6f} ICIR={ir:.6f} hit={hit:.4f} dates={n} mean_valid={nv:.2f}')
 if h==5:
  for name,mask in [('2020',q.date.dt.year==2020),('2021_22',q.date.dt.year.isin([2021,2022])),('2023_24',q.date.dt.year.isin([2023,2024])),('2025_26',q.date.dt.year>=2025)]:
   s=q.loc[mask,'ic']; print(name, 'IC=%.6f ICIR=%.6f dates=%d'%(s.mean(),s.mean()/s.std(ddof=1),len(s)))
# turnover from date-aligned cross-sectional ranks
wide=pd.DataFrame(signals).sort_index(); rank=wide.rank(axis=1,pct=True)
tv=[]
for i in range(1,len(rank)):
 c=rank.iloc[i].corr(rank.iloc[i-1],method='spearman')
 if pd.notna(c): tv.append(1-c)
print('mean rank turnover',np.mean(tv),'signal valid coverage',wide.notna().stack().mean())
# Library signals precisely reconstructed at common cells
lib={}
for a in ASSETS:
 d=pd.read_csv(os.path.join(base,a+'.csv'),parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; c=pd.Series(d.close.to_numpy(),index=d.date); r=c.pct_change()
 lib.setdefault('miner_3_risk_adjusted_trend_20d',{})[a]=(c/c.shift(20)-1)/r.rolling(20,min_periods=15).std()
 lib.setdefault('miner_1_ravmom_20obs',{})[a]=(c/c.shift(20)-1)/r.rolling(20,min_periods=15).std()
 lib.setdefault('miner_1_volnorm_reversal_5obs',{})[a]=-(c/c.shift(5)-1)/r.rolling(5,min_periods=4).std()
 lib.setdefault('miner_2_volscaled_reversal_1obs',{})[a]=-r/r.rolling(20,min_periods=15).std()
 lib.setdefault('miner_2_realized_volatility_20obs',{})[a]=r.rolling(20,min_periods=15).std()
 # volume factor approximation: relative 20d volume
 v=pd.Series(d.volume.to_numpy(),index=d.date); lib.setdefault('miner_3_relative_volume_participation_20d',{})[a]=v/v.rolling(20,min_periods=15).mean()
for name,dd in lib.items():
 x=wide.stack().rename('x'); y=pd.DataFrame(dd).stack().rename('y'); j=pd.concat([x,y],axis=1).dropna()
 print('LIBCORR',name,'rho=%.6f cells=%d'%(j.x.corr(j.y,method='spearman'),len(j)))
