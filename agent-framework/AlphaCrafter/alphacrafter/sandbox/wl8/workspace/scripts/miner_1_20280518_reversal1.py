import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-05-17'); rows=[]
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if os.path.exists(f):
  d=pd.read_csv(f,parse_dates=['date']).sort_values('date');d=d[d.date<=END];r=d.close.pct_change(); sig=(-r).shift(1); fw=d.close.shift(-1)/d.close-1
  rows += list(zip(d.date,[s]*len(d),sig,fw))
z=pd.DataFrame(rows,columns=['date','symbol','signal','fwd']).replace([np.inf,-np.inf],np.nan).dropna(); vals=[];meta=[]
for dt,g in z.groupby('date'):
 if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1: vals.append(spearmanr(g.signal,g.fwd).statistic);meta.append((dt,len(g)))
a=np.array(vals);print('dates',len(a),'rows',len(z),'avg_n',np.mean([n for _,n in meta]),'coverage',len(z)/(15*len(meta)),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'turnover',np.nan)
for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2026-12-31'),('2027','2027-12-31'),('2028','2028-05-17'),('2027-11-01','2028-05-17')]:
 q=np.array([v for (dt,n),v in zip(meta,a) if pd.Timestamp(lo)<=dt<=pd.Timestamp(hi)]);print(lo,len(q),q.mean(),q.mean()/q.std(ddof=1) if len(q)>1 else np.nan)
z[['date','symbol','signal']].to_csv('scripts/miner_1_20280518_reversal1_signal.csv',index=False)
