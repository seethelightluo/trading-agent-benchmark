import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
D={}
for s in U:
 p=Path('../persistent/stock_data')/(s+'.csv'); x=pd.read_csv(p); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index();
# Conditional volatility-scaled reversal: inverse 3d return, normalized by trailing 20d vol,
# then centered cross-section and scaled by contemporaneous cross-sectional dispersion.
ret3=R.rolling(3,min_periods=3).sum(); vol=R.rolling(20,min_periods=15).std(); raw=-ret3/(vol*np.sqrt(3)); F=raw.sub(raw.median(axis=1),axis=0)
disp=R.std(axis=1).rolling(20,min_periods=10).mean(); F=F.mul(disp,axis=0) # regime interaction, preserves rankings but tests conditional magnitude
for h in [1,5,10]:
 Y=R.shift(-1).rolling(h,min_periods=h).sum().shift(-(h-1)); vals=[]; ns=[]; dates=[]
 for dt in R.index:
  z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); dates.append(dt)
 q=pd.Series(vals,index=dates); print('h',h,'IC %.7f ICIR %.7f hit %.4f dates %d avgN %.2f coverage %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns),len(q)/len(R)))
 for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
  z=q.loc[a:b]; print(a+'-'+b,'IC %.6f ICIR %.6f n %d'%(z.mean(),z.mean()/z.std(ddof=1),len(z)))
# rank turnover
rank=F.rank(axis=1,pct=True); turn=(rank-rank.shift(1)).abs().mean(axis=1).dropna(); print('turnover',turn.mean(),'valid instruments',F.notna().mean().mean())
# signal artifact
out=F.copy(); out.index.name='date'; out.reset_index().to_csv('scripts/miner_2_20261217_conditional_vol_reversal_signal.csv',index=False)
