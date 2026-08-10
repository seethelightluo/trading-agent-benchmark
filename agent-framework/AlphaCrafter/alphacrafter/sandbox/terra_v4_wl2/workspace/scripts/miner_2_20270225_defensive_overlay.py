import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; EQ=U[:8]; DEF=['XAU','US10Y','CN10Y']
def load(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   x=fn(s,days=5000)
   if x is not None and len(x): return x
  except Exception: pass
px=pd.concat({s:load(s).set_index('date')['close'] for s in U},axis=1).sort_index(); r=px.pct_change()
trend=r.rolling(20,min_periods=20).sum(); vol=r.rolling(20,min_periods=20).std()
rank=trend.rank(axis=1,pct=True); medvol=vol.median(axis=1)
z=(rank-.5).div(vol.div(medvol,axis=0).replace(0,np.nan))
breadth=r[EQ].lt(0).sum(axis=1)/r[EQ].notna().sum(axis=1); stress=breadth.shift(1)>=.625
sig=z.copy(); sig.loc[stress,DEF]=sig.loc[stress,DEF]+0.75
sig=sig.rank(axis=1,pct=True); sig=sig.sub(sig.mean(axis=1),axis=0)
for h in [1,5,10]:
 fwd=px.shift(-h)/px-1; vals=[]; ns=[]; dates=[]
 for d in sig.index:
  q=pd.concat([sig.loc[d],fwd.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:
   vals.append(q.iloc[:,0].corr(q.iloc[:,1],method='spearman')); ns.append(len(q));dates.append(d)
 a=np.array(vals); print('h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4),'matrix_cov',round(sig.notna().sum().sum()/(len(U)*len(sig)),4))
 if h==10:
  for name,mask in [('stress',stress),('normal',~stress)]:
   aa=[vals[i] for i,d in enumerate(dates) if bool(mask.loc[d])]; print(name,'dates',len(aa),'IC',round(np.mean(aa),6) if aa else None,'ICIR',round(np.mean(aa)/np.std(aa,ddof=1),6) if len(aa)>1 else None)
sig.index.name='date'; sig.to_csv('../persistent/factor_signals_miner_2_20270225_defensive_overlay.csv')
