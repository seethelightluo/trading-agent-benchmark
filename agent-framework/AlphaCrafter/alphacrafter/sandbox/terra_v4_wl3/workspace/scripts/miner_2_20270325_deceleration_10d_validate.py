import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
end=pd.Timestamp('2027-03-25'); P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).set_index('date').sort_index()
 P[s]=d.close.loc[:end]
P=pd.DataFrame(P).ffill()
f=P.pct_change(20)/4-P.pct_change(5)
for h in (1,5,10):
 y=P.shift(-h)/P-1; rows=[]
 for dt in f.index:
  x=f.loc[dt]; z=y.loc[dt]; ok=x.notna()&z.notna()
  if ok.sum()>=8: rows.append((dt,spearmanr(x[ok],z[ok]).statistic,ok.sum()))
 q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 mean=q.ic.mean(); ir=mean/q.ic.std(ddof=1)*np.sqrt(252)
 print(f'h={h} dates={len(q)} avgN={q.n.mean():.4f} IC={mean:.10f} ICIR={ir:.10f} hit={(q.ic>0).mean():.10f}')
 print('regimes',[(a,b,len(q.loc[a:b]),q.loc[a:b].ic.mean()) for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2027-03-25')]])
print('coverage',f.notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
rows=[(dt,s,f.loc[dt,s]) for dt in f.index for s in U]
pd.DataFrame(rows,columns=['date','symbol','signal']).to_csv('scripts/miner_2_20270325_deceleration_raw_signal.csv',index=False)
