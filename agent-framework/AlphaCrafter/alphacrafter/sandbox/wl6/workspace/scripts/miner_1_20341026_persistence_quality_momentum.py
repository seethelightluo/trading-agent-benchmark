import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=4000)
   if d is not None and len(d): return d
  except Exception: pass
S={}
for s in U:
 d=fetch(s)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); S[s]=d.set_index('date')
px=pd.DataFrame({s:x.close.astype(float) for s,x in S.items()}).sort_index(); r=px.pct_change()
# Persistence-quality momentum: 40d return, multiplied by directional persistence
# (fraction of positive days minus 0.5), then scaled by 40d realized volatility.
# One-day lag ensures only completed data are used.
ret=px/px.shift(40)-1
persist=(r.gt(0).rolling(40,min_periods=28).mean()-0.5)
vol=r.rolling(40,min_periods=28).std()
factor=(ret*persist/(vol*np.sqrt(40))).shift(1)
factor=factor.sub(factor.median(axis=1),axis=0)
factor.to_csv('scripts/miner_1_20341026_persistence_quality_momentum_signal.csv',index_label='date')
print('assets_loaded',px.shape[1],'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20,40]:
 fw=px.shift(-h)/px-1; rows=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1])
   if np.isfinite(c): rows.append((dt,c,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); a=q.ic.to_numpy(); m=a.mean()
 print('h',h,'dates',len(a),'avg_names',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,4),'IC',round(m,8),'ICIR',round(m/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),4))
 for label,lo,hi in [('early','2020-01-01','2027-12-31'),('mid','2028-01-01','2031-12-31'),('recent','2032-01-01','2034-10-11')]:
  b=q[(q.date>=lo)&(q.date<=hi)].ic
  if len(b)>20: print(' regime',label,'n',len(b),'IC',round(b.mean(),6),'hit',round((b>0).mean(),3))
print('turnover',round(factor.rank(pct=True).diff().abs().stack().mean(),6))
