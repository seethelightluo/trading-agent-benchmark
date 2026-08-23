import pandas as pd,numpy as np
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2029-09-06')
p={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].sort_index();p[a]=d
p=pd.DataFrame(p).loc[:END]; r=np.log(p).diff(); v=r.rolling(30).std(); shock=r.rolling(3).sum()/(v*np.sqrt(3)+.02); rec=-shock # reversal after recent shock
# exclude cross-sectional date mean to make relative
f=rec.sub(rec.mean(axis=1),axis=0); y=np.log(p.shift(-10)/p);ics=[];ds=[];ns=[]
for d in f.index:
 ok=f.loc[d].notna()&y.loc[d].notna()
 if ok.sum()>=8:
  q=spearmanr(f.loc[d,ok],y.loc[d,ok]).statistic
  if np.isfinite(q):ics.append(q);ds.append(d);ns.append(ok.sum())
a=np.array(ics);print('dates',len(a),'range',ds[0].date(),ds[-1].date(),'avg_n',np.mean(ns),'coverage',np.mean(ns)/15);print('IC',a.mean(),'ICIR',a.mean()/a.std(ddof=1)*np.sqrt(252/10),'hit',np.mean(a>0),'turn',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lo,hi in [(2020,2022),(2023,2024),(2025,2026),(2027,2028),(2029,2029)]:
 q=a[[lo<=d.year<=hi for d in ds]];print(lo,hi,len(q),q.mean() if len(q) else np.nan)
