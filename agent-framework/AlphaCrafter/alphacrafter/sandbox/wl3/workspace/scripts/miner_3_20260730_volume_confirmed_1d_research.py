import pandas as pd,numpy as np
from scipy.stats import spearmanr
S=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2026-07-15'); sig={}; ret={}
for s in S:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv');d.date=pd.to_datetime(d.date);d=d[d.date<=cut].sort_values('date').set_index('date');r=d.close.pct_change(fill_method=None);rv=r.rolling(20,min_periods=15).std();vs=d.volume/d.volume.rolling(20,min_periods=15).median();sig[s]=(-r/rv)*np.sqrt(vs.clip(.5,3));ret[s]=r
sig=pd.DataFrame(sig); ret=pd.DataFrame(ret); fwd=ret.shift(-1); rows=[]; tr=[]; prev=None
for dt in sig.index.union(fwd.index):
 x=sig.loc[dt] if dt in sig.index else pd.Series(dtype=float); y=fwd.loc[dt] if dt in fwd.index else pd.Series(dtype=float);z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  rows.append((dt,spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic,len(z)));rank=x.rank(pct=True)
  if prev is not None:tr.append((rank-prev).abs().mean())
  prev=rank
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
def p(a):return len(a),a.ic.mean(),a.ic.std(ddof=1),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean(),a.n.mean()
print('overall',p(q));
for n,a,b in [('2020-22','2020-01-01','2022-12-31'),('2023-24','2023-01-01','2024-12-31'),('2025-26','2025-01-01','2026-07-15')]:print(n,p(q.loc[a:b]))
print('coverage',sig.notna().sum(axis=1).mean()/15,'turnover',np.nanmean(tr))
for h in [5,10]:
 yy=ret.copy(); yy={s:ret[s].rolling(h).sum().shift(-h+1) for s in S};yy=pd.DataFrame(yy);a=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=pd.Series(a);print('decay',h,len(a),a.mean(),a.mean()/a.std(ddof=1))
q.to_csv('scripts/miner_3_20260730_volume_confirmed_1d_research.csv')
