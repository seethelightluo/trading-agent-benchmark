import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   d=f(s,days=5000)
   if d is not None and len(d): return d
  except Exception: pass
S={}
for s in U:
 d=fetch(s)
 if d is not None:
  d=d.copy();d.date=pd.to_datetime(d.date);S[s]=d.set_index('date')
px=pd.DataFrame({s:x.close.astype(float) for s,x in S.items()}).sort_index(); r=px.pct_change()
# Defensive low-volatility carry: inverse realized volatility, combined with mild positive short-term carry.
rv=r.rolling(30,min_periods=20).std()*np.sqrt(252)
carry=px.pct_change(10)
factor=((-rv.rank(axis=1,pct=True)) + .20*carry.rank(axis=1,pct=True)).shift(1)
factor.to_csv('scripts/miner_1_20340817_lowvol_carry_signal.csv',index_label='date')
print('assets_loaded',px.shape[1],'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20,40]:
 fw=px.shift(-h)/px-1; rows=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1])
   if np.isfinite(c): rows.append((dt,c,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']);a=q.ic.to_numpy();m=a.mean()
 print('h',h,'dates',len(a),'avg_names',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,4),'IC',round(m,8),'ICIR',round(m/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),4))
 for label,lo,hi in [('early','2020','2026'),('mid','2026','2030'),('recent','2030','2035')]:
  b=q[(q.date>=lo)&(q.date<hi)].ic
  if len(b)>20: print('regime',label,'h',h,'n',len(b),'IC',round(b.mean(),8),'hit',round((b>0).mean(),4))
print('turnover',round(factor.rank(pct=True).diff().abs().stack().mean(),6))
