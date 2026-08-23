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
v=pd.read_csv('../persistent/index_data/VIX.csv'); v['date']=pd.to_datetime(v['date']); vv=v.set_index('date').close.astype(float).reindex(px.index).ffill()
rr=px.pct_change(5); rel=rr.sub(rr.median(axis=1),axis=0); rv=r.rolling(20,min_periods=15).std(); z=(rel/(rv*np.sqrt(5))).clip(-3,3)
vp=vv.rolling(252,min_periods=126).rank(pct=True); gate=(0.5+vp).clip(.5,1.5); factor=z.mul(-gate,axis=0).shift(1)
factor.to_csv('scripts/miner_2_20340831_vix_gated_reversal_signal.csv',index_label='date')
print('assets_loaded',px.shape[1],'dates',len(px),'cutoff',px.index.max().date(),'valid_factor',int(factor.notna().sum().sum()))
for h in [5,10,20]:
 fw=px.shift(-h)/px-1; rows=[]
 for dt in factor.index:
  q=pd.concat([factor.loc[dt],fw.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   c=q.iloc[:,0].corr(q.iloc[:,1])
   if np.isfinite(c): rows.append((dt,c,len(q)))
 q=pd.DataFrame(rows,columns=['date','ic','n']); a=q.ic.to_numpy()
 if len(a)<2: print('h',h,'NO_VALID_DATES'); continue
 print('h',h,'dates',len(a),'avg_names',round(q.n.mean(),3),'coverage',round(q.n.mean()/15,4),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),4))
 for label,lo,hi in [('early','2020','2025-12-31'),('mid','2026','2029-12-31'),('late','2030','2034-12-31')]:
  x=q[(q.date>=pd.Timestamp(lo))&(q.date<=pd.Timestamp(hi))].ic
  if len(x): print('regime',label,'n',len(x),'IC',round(x.mean(),8))
print('turnover',round(factor.rank(pct=True).diff().abs().stack().mean(),6))