import pandas as pd,numpy as np
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; E=pd.Timestamp('2035-01-03')
def rd(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index(); return pd.to_numeric(d.loc[d.index<=E,'close'],errors='coerce')
P=pd.DataFrame({a:rd(a) for a in A}); R=P.pct_change(fill_method=None)
# Short-horizon mean reversion: recent 5-day loss, normalized by trailing 20-day risk;
# cross-sectional demean prevents broad-market direction from entering the signal.
vol=R.rolling(20,min_periods=15).std(); raw=-(P/P.shift(5)-1)/(vol*np.sqrt(5)+1e-12); F=raw.sub(raw.mean(axis=1),axis=0)
def calc(h, lo=None, hi=None):
 x=[]
 for i in range(len(P)-h):
  if lo is not None and not (P.index[i]>=pd.Timestamp(lo) and P.index[i]<=pd.Timestamp(hi)): continue
  q=pd.concat([F.iloc[i].rename('f'),(P.iloc[i+h]/P.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(q)>=8: x.append(q.f.corr(q.y,method='spearman'))
 s=pd.Series(x).dropna(); return len(s),float(s.mean()),float(s.mean()/(s.std(ddof=1)+1e-12)),float((s>0).mean())
print('cutoff',E.date(),'dates',len(P),'assets',len(A),'validcells',int(F.count().sum()),'coverage',float(F.count().sum()/(len(P)*len(A))))
for h in [1,5,10,20]: print('H',h,calc(h))
for label,lo,hi in [('2020-25','2020','2025-12-31'),('2026-29','2026','2029-12-31'),('2030-32','2030','2032-12-31'),('2033-35','2033','2035-01-03')]: print(label,calc(10,lo,hi))
# rank turnover and decay by raw rank
r=F.rank(axis=1,pct=True); print('turnover',float(r.diff().abs().mean(axis=1).dropna().mean()))
# independent simple proxy screen (not admission evidence)
for n,z in [('neg5',-(P/P.shift(5)-1)),('neg1',-R),('neg20',-(P/P.shift(20)-1)/vol)]:
 q=pd.concat([F.stack().rename('f'),z.stack().rename('z')],axis=1).dropna(); print('proxy',n,float(q.f.corr(q.z,method='spearman')))
