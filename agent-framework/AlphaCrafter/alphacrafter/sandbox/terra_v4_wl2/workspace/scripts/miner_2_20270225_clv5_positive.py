import pandas as pd,numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; D={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date'); clv=(d.close-d.low)/(d.high-d.low).replace(0,np.nan); D[s]=pd.DataFrame({'sig':clv.rolling(5,min_periods=4).mean(),'close':d.close})
all_dates=sorted(set().union(*[set(x.index[:-1]) for x in D.values()])); rows=[]; ics=[]
for dt in all_dates:
 vals=[]; fw=[]
 for s in U:
  x=D[s]
  if dt in x.index:
   p=x.index.get_loc(dt); q=p+1
   if q<len(x) and pd.notna(x.iloc[p].sig): vals.append(x.iloc[p].sig); fw.append(x.iloc[q].close/x.iloc[p].close-1)
 if len(vals)>=8 and np.std(vals)>0 and np.std(fw)>0: rows.append((dt,len(vals))); ics.append(spearmanr(vals,fw).statistic)
a=np.array(ics); print('factor=positive 5d avg CLV; dates',len(a),'avg_names',np.mean([n for _,n in rows]),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0),'coverage',np.mean([n for _,n in rows])/15)
for lo,hi in [('2020-01-01','2022-12-31'),('2023-01-01','2024-12-31'),('2025-01-01','2026-12-31'),('2027-01-01','2027-02-25')]:
 z=np.array([v for (dt,_),v in zip(rows,a) if lo<=str(dt.date())<=hi]); print(lo,hi,len(z),z.mean() if len(z) else np.nan,(z.mean()/z.std(ddof=1)) if len(z)>1 else np.nan)
out=[]
for s in U:
 for dt,v in D[s].sig.items(): out.append([dt,s,v])
pd.DataFrame(out,columns=['date','symbol','signal']).to_csv('../persistent/factor_signals_miner_2_20270225_clv5_reversal.csv',index=False)
