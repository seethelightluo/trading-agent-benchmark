import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def fetch(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=4000)
   if d is not None and len(d): return d
  except Exception: pass
S={}
for s in U:
 d=fetch(s)
 if d is not None:
  d=d.copy(); d.date=pd.to_datetime(d.date); S[s]=d.set_index('date')
px=pd.DataFrame({s:x.close.astype(float) for s,x in S.items()}).sort_index(); r=px.pct_change()
# Contrarian short shock, stronger when cross-asset dispersion is elevated; normalize by own risk.
ret5=px/px.shift(5)-1
vol20=r.rolling(20,min_periods=15).std()
disp=r.rolling(5,min_periods=4).std().median(axis=1)
base=-(ret5/(vol20+1e-12))
threshold=disp.rolling(60,min_periods=40).median()
f=base.mul((disp/(threshold+1e-12)).clip(0.75,1.75),axis=0)
f=f.shift(1).sub(f.shift(1).median(axis=1),axis=0)
f.to_csv('scripts/miner_1_20341207_dispersion_shock_reversal_signal.csv',index_label='date')
print('assets_loaded',px.shape[1],'dates',len(px),'cutoff',px.index.max().date(),'finite',int(f.notna().sum().sum()))
for h in [5,10,20,40]:
 fw=px.shift(-h)/px-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1])
   if np.isfinite(c): rows.append((dt,c,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); a=q.ic.to_numpy(); m=a.mean() if len(a) else np.nan; ir=m/a.std(ddof=1)*np.sqrt(len(a)) if len(a)>1 and a.std(ddof=1)>0 else np.nan
 print('h',h,'dates',len(a),'avg_names',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,4),'IC',round(m,8),'ICIR',round(ir,6),'hit',round((a>0).mean(),4))
 for label,lo,hi in [('early','2020-01-01','2027-12-31'),('mid','2028-01-01','2031-12-31'),('recent','2032-01-01','2034-12-06')]:
  b=q[(q.date>=lo)&(q.date<=hi)].ic
  if len(b)>20: print(' regime',label,'n',len(b),'IC',round(b.mean(),6),'hit',round((b>0).mean(),3))
print('turnover',round(f.rank(pct=True).diff().abs().stack().mean(),6))
