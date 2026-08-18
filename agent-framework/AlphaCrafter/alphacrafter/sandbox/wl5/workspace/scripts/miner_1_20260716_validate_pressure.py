import pandas as pd, numpy as np
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; P={}
for s in U:
 d=pd.read_csv('../persistent/stock_data/'+s+'.csv'); d.date=pd.to_datetime(d.date); P[s]=d.set_index('date')
idx=pd.date_range('2020-01-01','2026-07-15',freq='D'); close=pd.DataFrame({s:P[s].close.reindex(idx) for s in U}); op=pd.DataFrame({s:P[s].open.reindex(idx) for s in U}); hi=pd.DataFrame({s:P[s].high.reindex(idx) for s in U}); lo=pd.DataFrame({s:P[s].low.reindex(idx) for s in U})
clv=2*(close-lo)/(hi-lo).replace(0,np.nan)-1; f=-clv
print('range',idx.min(),idx.max(),'dates',len(idx),'instruments',len(U))
def stats(h,mask=None):
 rows=[]
 for i in range(len(idx)-h):
  if mask is not None and not mask[i]: continue
  z=pd.concat([f.iloc[i].rename('f'),(close.iloc[i+h]/close.iloc[i]-1).rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1: rows.append(z.f.corr(z.r,method='spearman'))
 q=pd.Series(rows).dropna(); return len(q),q.mean(),q.mean()/q.std(ddof=1),(q>0).mean()
for h in [1,5,10]: print('DECAY',h,stats(h))
for lab,mask in [('2020-22',idx<'2023-01-01'),('2023-24',((idx>='2023-01-01')&(idx<'2025-01-01'))),('2025-26',idx>='2025-01-01')]: print('REGIME',lab,stats(1,mask))
rank=f.rank(axis=1,pct=True); print('turnover_rank',((rank.diff().abs()>0.2).sum(axis=1)/rank.notna().sum(axis=1)).mean())
rets=close.pct_change(); F=pd.DataFrame({'pressure':f.stack(),'reversal':(-rets.rolling(5).sum()).stack(),'ram':(rets.rolling(20).sum()/rets.rolling(20).std()).stack(),'peer':(rets.rolling(5).sum().sub(rets.rolling(5).sum().median(axis=1),axis=0)).stack()}).dropna(); print('CORR',F.corr().round(4).to_dict()['pressure']); print('coverage',f.notna().stack().mean())
