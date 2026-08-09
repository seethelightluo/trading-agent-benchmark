import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr
import numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv')); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').set_index('date'); D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index(); F=-R.rolling(5,min_periods=5).sum(); Y=R.shift(-1); vals=[]; ns=[]; ds=[]
for dt in R.index:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).dropna()
 if len(z)>=8: vals.append(spearmanr(z.f,z.y).statistic); ns.append(len(z)); ds.append(dt)
q=pd.Series(vals,index=ds); print(q.mean(),q.mean()/q.std(ddof=1),(q>0).mean(),len(q),np.mean(ns),len(q)/len(R))
F.index.name='date'; F.reset_index().to_csv('scripts/miner_2_20261217_reversal5_signal.csv',index=False)
