import pandas as pd, numpy as np
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).sort_values('date').set_index('date') for s in U}
rows=[]
for s,x in D.items():
 r=x.close.pct_change(); vol=r.rolling(20,min_periods=15).std()
 # recent 10-day trend scaled by trailing risk, strictly close-to-close
 f=r.rolling(10,min_periods=8).sum()/(vol*np.sqrt(10)+1e-12)
 y=x.close.shift(-10)/x.close-1
 for d in x.index:
  if pd.Timestamp('2020-01-01')<=d<=pd.Timestamp('2027-12-03') and pd.notna(f.loc[d]) and pd.notna(y.loc[d]): rows.append((d,s,float(f.loc[d]),float(y.loc[d])))
a=pd.DataFrame(rows,columns=['date','symbol','factor','fwd'])
a.to_csv('scripts/miner_3_20271203_risk_momentum_10d_signal.csv',index=False)
for label,lo,hi in [('all','2020-01-01','2027-12-03'),('online','2026-07-16','2027-12-03'),('recent','2027-01-01','2027-12-03'),('2020_22','2020-01-01','2022-12-31'),('2023_25','2023-01-01','2025-12-31')]:
 q=a[(a.date>=lo)&(a.date<=hi)]; vals=[]; ns=[]
 for d,g in q.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>1 and g.fwd.nunique()>1:
   vals.append(spearmanr(g.factor,g.fwd).statistic);ns.append(len(g))
 z=np.array(vals); ic=np.nanmean(z); ir=ic/np.nanstd(z,ddof=1)
 print(label,'dates',len(z),'avgN',round(np.mean(ns),2),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round(np.mean(z>0),4),'coverage',round(q.symbol.nunique()/15,4))
# decay
for h in [1,5,10,20]:
 vals=[]
 for s,x in D.items():
  r=x.close.pct_change(); vol=r.rolling(20,min_periods=15).std(); f=r.rolling(10,min_periods=8).sum()/(vol*np.sqrt(10)+1e-12); y=x.close.shift(-h)/x.close-1
  z=pd.DataFrame({'f':f,'y':y}).dropna()
  for d,g in z.loc['2020-01-01':'2027-12-03'].groupby(level=0):
   if len(g)>=8 and g.f.nunique()>1 and g.y.nunique()>1: vals.append(spearmanr(g.f,g.y).statistic)
 print('decay',h,round(np.nanmean(vals),6),len(vals))
