import numpy as np, pandas as pd
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
  d=d.copy(); d.date=pd.to_datetime(d.date); S[s]=d.set_index('date').sort_index().close.astype(float)
px=pd.DataFrame(S).sort_index(); r=px.pct_change(); r10=px.pct_change(10)
# downside semideviation uses zeros on non-down days, avoiding sparse-NaN rolling failure.
neg2=(r.clip(upper=0)**2).rolling(30,min_periods=20).mean()
down=np.sqrt(neg2)*np.sqrt(30)
factor=(-r10/(down+1e-8)).shift(1)
factor.to_csv('scripts/miner_1_20340608_downside_reversal_10d_signal.csv',index_label='date')
print('assets_loaded',px.shape[1],'rows',len(px),'valid',int(factor.notna().sum().sum()),flush=True)
for h in [5,10,20,40]:
 fw=px.shift(-h)/px-1; rows=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1])
   if np.isfinite(c): rows.append((dt,c,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); a=q.ic.to_numpy(); m=a.mean(); ir=m/a.std(ddof=1)*np.sqrt(len(a))
 print('h',h,'dates',len(a),'avg_names',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,4),'IC',round(m,8),'ICIR',round(ir,6),'hit',round((a>0).mean(),4),flush=True)
 if h==10:
  for aa,bb in [('2020','2024'),('2025','2026'),('2027','2029'),('2030','2032'),('2033','2034')]:
   x=q[(q.date.astype(str)>=aa)&(q.date.astype(str)<=bb)].ic; print('regime',aa,bb,'dates',len(x),'IC',round(x.mean(),8) if len(x) else None)
print('turnover',factor.rank(pct=True).diff().abs().stack().mean(),flush=True)
