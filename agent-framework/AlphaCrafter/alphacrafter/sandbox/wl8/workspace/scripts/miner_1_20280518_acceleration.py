import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-05-17')
rows=[]
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f): continue
 d=pd.read_csv(f,parse_dates=['date']).sort_values('date'); d=d[d.date<=END]; r=d.close.pct_change();
 # trend acceleration: recent 5d return minus average of prior 15d, volatility normalized, all lagged
 sig=((r.rolling(5).sum()-r.shift(5).rolling(15).sum()/3)/r.rolling(20).std()).shift(1)
 for h in [1,3,5]:
  fw=d.close.shift(-h)/d.close-1
  rows += list(zip(d.date,[s]*len(d),[h]*len(d),sig,fw))
z=pd.DataFrame(rows,columns=['date','symbol','h','signal','fwd']).replace([np.inf,-np.inf],np.nan).dropna()
for h in [1,3,5]:
 x=z[z.h==h]; vals=[]; meta=[]
 for dt,g in x.groupby('date'):
  if len(g)>=8 and g.signal.nunique()>1 and g.fwd.nunique()>1: vals.append(spearmanr(g.signal,g.fwd).statistic);meta.append((dt,len(g)))
 a=np.array(vals); print('H',h,'dates',len(a),'rows',len(x),'avg_n',np.mean([n for _,n in meta]),'coverage',len(x)/(15*len(meta)),'IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1),'hit',np.mean(a>0))
 for lo,hi in [('2020','2022-12-31'),('2023','2025-12-31'),('2026','2026-12-31'),('2027','2027-12-31'),('2028','2028-05-17'),('2027-11-01','2028-05-17')]:
  q=np.array([v for (dt,n),v in zip(meta,a) if pd.Timestamp(lo)<=dt<=pd.Timestamp(hi)]); print(lo,len(q),round(q.mean(),6) if len(q) else np.nan,round(q.mean()/q.std(ddof=1),6) if len(q)>1 else np.nan)
 if h==1: x[['date','symbol','signal']].to_csv('scripts/miner_1_20280518_accel_signal.csv',index=False)
