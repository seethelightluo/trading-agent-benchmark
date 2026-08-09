import pandas as pd, numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cutoff=pd.Timestamp('2027-02-24')
rows=[]; sig=[]
for s in A:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date'); d=d[d.date<=cutoff].reset_index(drop=True)
 f=-(d.close/d.open-1).rolling(3,min_periods=3).mean()
 y=d.close.shift(-1)/d.close-1
 for i in range(len(d)-1):
  if np.isfinite(f.iloc[i]) and np.isfinite(y.iloc[i]): rows.append((d.date.iloc[i],s,f.iloc[i],y.iloc[i]))
 sig += [(d.date.iloc[i],s,f.iloc[i]) for i in range(len(d)) if np.isfinite(f.iloc[i])]
x=pd.DataFrame(rows,columns=['date','asset','factor','fwd']); wide=x.pivot(index='date',columns='asset',values=['factor','fwd'])
ics=[]
for dt,g in x.groupby('date'):
 if len(g)>=8: ics.append((dt,spearmanr(g.factor,g.fwd).statistic,len(g)))
z=pd.DataFrame(ics,columns=['date','ic','n']); ic=z.ic.mean(); ir=ic/z.ic.std(ddof=1)
print('dates',len(z),'avg_n',z.n.mean(),'IC',ic,'ICIR',ir,'hit',(z.ic>0).mean(),'coverage',len(x)/(len(z)*15))
for a,b in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-06-30'),('2026-07-01','2027-02-24')]:
 q=z[(z.date>=a)&(z.date<=b)].ic; print(a,b,len(q),q.mean(),q.mean()/q.std(ddof=1))
# rank turnover on complete signal panels
p=x.pivot(index='date',columns='asset',values='factor').rank(axis=1,pct=True); print('turnover',p.diff().abs().mean(axis=1).mean())
out=pd.DataFrame(sig,columns=['date','asset','signal']); out.to_csv('../persistent/factor_signals_miner_1_20270225_intraday_reversal3.csv',index=False); print('artifact',len(out))
