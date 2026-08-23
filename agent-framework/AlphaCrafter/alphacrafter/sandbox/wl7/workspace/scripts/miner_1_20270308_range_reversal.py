import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
# Short-horizon reversal scaled by recent range and volatility; raw per-asset calendars, no forward fill.
F={};Y={}
for s in U:
 f='../persistent/stock_data/'+s+'.csv'
 if not os.path.exists(f):continue
 d=pd.read_csv(f,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); c=d.close
 ret=c.pct_change(); vol=ret.shift(1).rolling(20,min_periods=15).std(); rng=((d.high-d.low)/c).shift(1).rolling(20,min_periods=15).mean()
 F[s]=(-ret.shift(1).rolling(3,min_periods=3).sum()/(vol*np.sqrt(3)))/(1+rng.clip(lower=0))
 Y[s]=c.pct_change().shift(-1)
dates=sorted(set().union(*[set(x.index) for x in F.values()])); rows=[];art=[]
for dt in dates:
 vals=[]; fw=[]
 for s in F:
  if dt in F[s].index and dt in Y[s].index:
   vals.append(F[s].loc[dt]);fw.append(Y[s].loc[dt])
 z=pd.DataFrame({'f':vals,'y':fw}).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.f,z.y).statistic,len(z)))
  for s in F:
   if dt in F[s].index and pd.notna(F[s].loc[dt]):art.append((dt,s,F[s].loc[dt]))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date');q=a[(a.index>=pd.Timestamp('2025-01-01'))&(a.index<=pd.Timestamp('2027-03-08'))]
mean=q.ic.mean(); sd=q.ic.std(ddof=1)
print('dates',len(q),'avg_n',q.n.mean(),'IC',mean,'ICIR',mean/sd*np.sqrt(252),'hit',(q.ic>0).mean(),'coverage',q.n.mean()/15,'early_late',q.iloc[:len(q)//2].ic.mean(),q.iloc[len(q)//2:].ic.mean())
for h in [1,5,10,20]:
 vals=[]
 for dt in dates:
  a1=[];a2=[]
  for s in F:
   if dt in F[s].index:
    ix=Y[s].index.get_loc(dt); future=Y[s].iloc[ix:ix+h].add(1).prod()-1 if ix+h<=len(Y[s]) else np.nan
    a1.append(F[s].loc[dt]);a2.append(future)
  z=pd.DataFrame({'f':a1,'y':a2}).dropna()
  if len(z)>=8:vals.append(spearmanr(z.f,z.y).statistic)
 print('decay',h,np.nanmean(vals),len(vals))
pd.DataFrame(art,columns=['date','symbol','signal']).to_csv('scripts/miner_1_20270308_range_reversal_signal.csv',index=False)
