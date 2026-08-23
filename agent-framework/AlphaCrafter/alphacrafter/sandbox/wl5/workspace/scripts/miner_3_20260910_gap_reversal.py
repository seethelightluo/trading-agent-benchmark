import pandas as pd,numpy as np,warnings
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-09-10')
D={s:pd.read_csv('../persistent/stock_data/'+s+'.csv',parse_dates=['date']).query('date<=@cut').set_index('date').sort_index() for s in U}
dates=pd.DatetimeIndex(sorted(set().union(*[set(x.index) for x in D.values()]))); P=pd.DataFrame({s:x.close for s,x in D.items()}).reindex(dates); O=pd.DataFrame({s:x.open for s,x in D.items()}).reindex(dates); R=P.pct_change()
for w in [1,3,5,10]:
 f=-(O/P.shift(1)-1).rolling(w,min_periods=w).mean(); ics=[];ns=[];ds=[]; rank=[]
 for i in range(len(dates)-1):
  q=pd.concat([f.iloc[i].rename('f'),R.iloc[i+1].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): ics.append(z);ns.append(len(q));ds.append(dates[i])
  rank.append(f.iloc[i].rank(pct=True))
 rank=pd.DataFrame(rank); turns=rank.diff().abs().mean(axis=1)
 x=np.array(ics); print('w',w,'dates',len(x),'N',round(np.mean(ns),2),'coverage',round(np.mean(ns)/15,4),'IC',round(x.mean(),5),'ICIR',round(x.mean()/x.std(ddof=1),5),'hit',round(np.mean(x>0),4),'turn',round(turns.mean(),4))
 for lo,hi in [(2020,2022),(2023,2024),(2025,2026)]:
  z=x[(pd.DatetimeIndex(ds).year>=lo)&(pd.DatetimeIndex(ds).year<=hi)];print(' regime',lo,hi,'IC',round(z.mean(),5),'n',len(z))
 for h in [5,10]:
  yy=P.pct_change(h).shift(-h); a=[]
  for i in range(len(dates)-h):
   q=pd.concat([f.iloc[i],yy.iloc[i].rename('y')],axis=1).dropna()
   if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.y.nunique()>1:
    z=spearmanr(q.iloc[:,0],q.y).statistic
    if np.isfinite(z):a.append(z)
  a=np.array(a); print(' horizon',h,'IC',round(a.mean(),5),'ICIR',round(a.mean()/a.std(ddof=1),5),'n',len(a))
f=-(O/P.shift(1)-1).rolling(3,min_periods=3).mean(); f.iloc[-1].to_csv('scripts/miner_3_20260910_gap3_signal.csv')
