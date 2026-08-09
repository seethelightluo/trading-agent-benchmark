import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17')
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index();
# One interpretable idea: medium-term momentum scaled by realized volatility.
for w in [10,20,40]:
 F=R.rolling(w,min_periods=w).sum()/R.rolling(w,min_periods=w).std()
 for h in [1,5,10]:
  Y=R.shift(-1).rolling(h).sum().shift(-(h-1)); vals=[]; ns=[]; dates=[]
  for dt in R.index:
   z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
   if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); dates.append(dt)
  q=pd.Series(vals,index=dates); print('w,h',w,h,'IC %.6f ICIR %.6f hit %.4f dates %d avgN %.2f cov %.4f'%(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns),np.mean(ns)/15))
  for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
   z=q[(q.index>=a)&(q.index<=b+'-12-31')]; print(' regime',a,b,'IC %.6f ICIR %.6f n %d'%(z.mean(),z.mean()/z.std(ddof=1),len(z)))
# rank turnover proxy for w=20
F=R.rolling(20,min_periods=20).sum()/R.rolling(20,min_periods=20).std(); ranks=F.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean().mean())
# signal artifact
F.to_csv('scripts/miner_1_20261217_volscaled_momentum_signal.csv')
