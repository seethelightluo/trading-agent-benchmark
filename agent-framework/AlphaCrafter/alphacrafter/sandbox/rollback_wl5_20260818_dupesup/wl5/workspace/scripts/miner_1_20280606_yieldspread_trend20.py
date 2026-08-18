import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
p=pd.DataFrame({a:pd.read_csv(os.path.join(base,a+'.csv'),parse_dates=['date']).set_index('date')['close'].sort_index() for a in assets}).sort_index()
# Yield-spread-conditioned medium-term trend: align 20d asset momentum with causal US10Y-CN10Y spread direction.
y10=p['US10Y'].pct_change(20); c10=p['CN10Y'].pct_change(20)
spread_signal=(y10-c10).rolling(5).mean()
r20=p.pct_change(20)
f=r20.mul(np.where(spread_signal>=0,1.0,-1.0),axis=0)
fr=p.shift(-10)/p-1
rows=[]
for dt in p.index:
 x=f.loc[dt]; y=fr.loc[dt]; ok=x.notna()&y.notna()
 if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],y[ok]).statistic,ok.mean(),ok.sum()))
z=pd.DataFrame(rows,columns=['date','ic','coverage','n']).set_index('date').replace([np.inf,-np.inf],np.nan).dropna()
mean=z.ic.mean(); sd=z.ic.std(ddof=1)
ranks=f.rank(axis=1,pct=True); to=(ranks-ranks.shift()).abs().mean(axis=1).mean()
print('dates',len(z),'avgN',z.n.mean(),'IC',mean,'ICIR',mean/sd*np.sqrt(252),'hit',(z.ic>0).mean(),'coverage',z.coverage.mean(),'turnover',to)
for label,lo,hi in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-12-31'),('2027-28','2027-01-01','2028-06-05')]:
 q=z.loc[lo:hi].ic; print(label,'n',len(q),'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1)*np.sqrt(252) if len(q)>1 else np.nan,'hit',(q>0).mean())
f.index.name='date'; f.to_csv('scripts/miner_1_20280606_yieldspread_trend20_signal.csv')
