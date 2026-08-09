import pandas as pd,numpy as np
from pathlib import Path
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-12-17'); D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'));x.date=pd.to_datetime(x.date);x=x[(x.date>=pd.Timestamp('2020-01-01'))&(x.date<=cut)].sort_values('date').set_index('date');D[s]=x.close.astype(float).pct_change()
R=pd.concat(D,axis=1).sort_index(); F=-R.rolling(3,min_periods=3).sum(); Y=R.shift(-1); rows=[]
for dt in R.index:
 z=pd.concat([F.loc[dt].rename('f'),Y.loc[dt].rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');print('dates',len(q),'range',q.index.min(),q.index.max(),'avg_n',q.n.mean(),'coverage',q.n.mean()/15,'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for a,b in [('2020','2021'),('2022','2023'),('2024','2025'),('2026-01-01','2026-12-17')]:
 z=q.loc[a:b].ic;print(a,b,len(z),round(z.mean(),5) if len(z) else None,round(z.mean()/z.std(ddof=1),5) if len(z)>1 else None)
F.stack().rename('signal').reset_index().rename(columns={'level_0':'date','level_1':'symbol'}).to_csv('scripts/miner_2_20261217_short_reversal_signal.csv',index=False)
