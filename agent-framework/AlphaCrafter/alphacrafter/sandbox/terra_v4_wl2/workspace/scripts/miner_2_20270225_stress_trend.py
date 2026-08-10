import pandas as pd, numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for f in (get_index_daily_data,get_stock_daily_data):
  try:
   x=f(s,days=5000)
   if x is not None and len(x): return x
  except Exception: pass
px=pd.concat({s:get(s).set_index('date')['close'] for s in U},axis=1).sort_index()
r=px.pct_change(); vol=r.rolling(20,min_periods=15).std(); mom=r.rolling(10,min_periods=10).sum()
cs=mom.sub(mom.median(axis=1),axis=0)
breadth=(r[U[:8]].lt(0).sum(axis=1)/r[U[:8]].notna().sum(axis=1)).shift(1)
for th in [.50,.625,.75]:
 stress=pd.DataFrame(np.broadcast_to((breadth>=th).to_numpy()[:,None],(len(breadth),len(U))),index=px.index,columns=U)
 f=(cs/vol).where(stress,-cs/vol)
 for h in [1,3,5,10]:
  fr=px.shift(-h)/px-1; vals=[]; ns=[]; rows=[]
  for d in f.index:
   z=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
   if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
    vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));rows.append([d,*f.loc[d].reindex(U).values])
  a=np.array(vals); print('th',th,'h',h,'dates',len(a),'avgN',round(np.mean(ns),2),'IC',round(np.mean(a),6),'ICIR',round(np.mean(a)/np.std(a,ddof=1),6),'hit',round(np.mean(a>0),4),'cov',round(f.notna().sum().sum()/(len(U)*len(f)),4))
  if th==.625 and h==5: pd.DataFrame(rows,columns=['date']+U).to_csv('../persistent/factor_signals_miner_2_20270225_stress_trend.csv',index=False)
