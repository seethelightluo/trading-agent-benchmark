import pandas as pd,numpy as np,os
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; b='../persistent/stock_data'; d={}
for a in A:
 f=f'{b}/{a}.csv'
 if os.path.exists(f): d[a]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index()['close']
C=pd.DataFrame(d).sort_index(); r=C.pct_change(); rb=r.mean(axis=1)
# residual momentum: asset 20d return minus contemporaneous cross-asset average, volatility-scaled
res=(r-rb,)
raw=C.pct_change(20).sub(C.pct_change(20).mean(axis=1),axis=0)
vol=r.rolling(60,min_periods=30).std(); F=raw/vol
ics={h:[] for h in [1,5,10,20]}; turns=[]; prev=None; counts=[]
for t in F.index:
 x=F.loc[t]; counts.append(x.notna().sum())
 if prev is not None:
  z=x.rank(pct=True).dropna(); common=z.index.intersection(prev.index); 
  if len(common)>=8: turns.append((z[common]-prev[common]).abs().mean())
  prev=z
 for h in ics:
  y=C.shift(-h).loc[t]/C.loc[t]-1; ok=x.notna()&y.notna()
  if ok.sum()>=8:
   q=spearmanr(x[ok],y[ok]).statistic
   if np.isfinite(q):ics[h].append(q)
print('dates',len(F),'universe',len(d),'avg_valid',np.mean(counts),'coverage_8',np.mean(np.array(counts)>=8))
for h,v in ics.items():
 s=pd.Series(v); print('horizon',h,'n',len(s),'IC',s.mean(),'ICIR',s.mean()/s.std(ddof=1),'hit',(s>0).mean())
print('turnover',np.mean(turns))
