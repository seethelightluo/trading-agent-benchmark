import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index() for s in U}
P=pd.DataFrame({s:D[s].close.pct_change(5) for s in U}); F=-(P.sub(P.median(axis=1),axis=0)); Y=pd.DataFrame({s:D[s].close.shift(-5)/D[s].close-1 for s in U})
rows=[]
for d in F.index:
 g=pd.DataFrame({'f':F.loc[d],'y':Y.loc[d]}).dropna()
 if d>=pd.Timestamp('2020-01-01') and len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: rows.append((d,spearmanr(g.f,g.y).statistic,len(g)))
a=pd.DataFrame(rows,columns=['date','ic','n']); a.to_csv('scripts/miner_3_20271203_relative_reversal_5d_signal.csv',index=False)
for name,lo,hi in [('all','2020-01-01','2027-12-03'),('online','2026-07-16','2027-12-03'),('recent','2027-01-01','2027-12-03'),('2020_22','2020-01-01','2022-12-31'),('2023_25','2023-01-01','2025-12-31')]:
 z=a[(a.date>=lo)&(a.date<=hi)].ic; print(name,'dates',len(z),'avgN',round(a[(a.date>=lo)&(a.date<=hi)].n.mean(),2),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
print('coverage',round(P.loc['2020-01-01':'2027-12-03'].notna().mean().mean(),4))
