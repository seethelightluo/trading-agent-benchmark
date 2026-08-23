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
  x=d.copy(); x.date=pd.to_datetime(x.date); S[s]=x.set_index('date').close.astype(float)
px=pd.DataFrame(S).sort_index(); r=px.pct_change(); r5=px.pct_change(5)
vol30=r.rolling(30).std()*np.sqrt(30)
res=r5.sub(r5.median(axis=1),axis=0)
f=(-res/vol30).shift(1)
f.to_csv('scripts/miner_2_20340525_volnorm_residual_reversal_30d_signal.csv',index_label='date')
print('assets_loaded',px.shape[1],'dates',len(px),'cutoff',px.index.max().date())
for h in [5,10,20,40]:
 fw=px.shift(-h)/px-1; rows=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1])
   if np.isfinite(c): rows.append((dt,c,len(z)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); a=q.ic.to_numpy(); m=a.mean(); ir=m/a.std(ddof=1)*np.sqrt(len(a))
 print('h',h,'dates',len(a),'avg_names',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,4),'IC',round(m,8),'ICIR',round(ir,6),'hit',round((a>0).mean(),4))
 if h==5:
  for aa,bb in [('2020','2024'),('2025','2029'),('2030','2032'),('2033','2034')]:
   x=q[(q.date.astype(str)>=aa)&(q.date.astype(str)<=bb)].ic
   print('regime',aa,bb,len(x),round(x.mean(),8) if len(x) else None)
print('turnover',round(f.rank(pct=True).diff().abs().stack().mean(),6))
