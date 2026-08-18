import os,numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').close for a in A}
p=pd.DataFrame(p).sort_index(); dxy=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.reindex(p.index).ffill()
# DXY-conditioned short-horizon cross-sectional reversal: reverse 3d return normally,
# but use momentum when dollar has fallen over 20d (macro liquidity regime).
r3=p.pct_change(3); regime=np.where(dxy.pct_change(20)<0,1.,-1.)
f=-r3.mul(regime,axis=0); fr=p.shift(-10)/p-1
rows=[]
for dt in p.index:
 x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.mean(),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','coverage','n']).set_index('date').dropna(); m=z.ic.mean(); sd=z.ic.std(ddof=1)
rank=f.rank(axis=1,pct=True); turn=(rank-rank.shift()).abs().mean(axis=1).dropna().mean()
print('dates',len(z),'avgN',z.n.mean(),'IC',m,'ICIR',m/sd,'hit',(z.ic>0).mean(),'coverage',z.coverage.mean(),'turnover',turn)
for label,lo,hi in [('2020-22','2020','2022'),('2023-24','2023','2024'),('2025-26','2025','2026'),('2027-28','2027','2028-05-08')]:
 q=z.loc[lo:hi].ic; print(label,len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean())
f.index.name='date';f.to_csv('scripts/miner_1_20280509_dxy_conditioned_reversal_signal.csv')
