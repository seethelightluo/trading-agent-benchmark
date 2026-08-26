import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={}
for s in U:
 d=None
 for f in (get_index_daily_data,get_stock_daily_data):
  try: d=f(s,4200)
  except Exception: d=None
  if d is not None: break
 if d is not None and len(d)>100:
  d=d.copy();d.date=pd.to_datetime(d.date);P[s]=d.set_index('date').sort_index()
cl=pd.DataFrame({s:d.close for s,d in P.items()}); hi=pd.DataFrame({s:d.high for s,d in P.items()});lo=pd.DataFrame({s:d.low for s,d in P.items()});vo=pd.DataFrame({s:d.volume for s,d in P.items()})
r=cl.pct_change(); shock=r.rolling(3,min_periods=3).sum().shift(1)
tr=(hi-lo)/cl
rz=((tr-tr.rolling(20,min_periods=10).median())/(tr.rolling(20,min_periods=10).std()+1e-12)).clip(0,2).shift(1)
lv=np.log(vo.replace(0,np.nan)); vz=((lv-lv.rolling(20,min_periods=10).median())/(lv.rolling(20,min_periods=10).std()+1e-12)).clip(0,2).shift(1)
# volume-confirmed range shock reversal; all inputs lagged one session
sig=(-shock*(1+rz)*(1+vz)).sub((-shock*(1+rz)*(1+vz)).median(axis=1),axis=0)
rows=[]
for dt in sig.index:
 y=cl.shift(-10).loc[dt]/cl.shift(-1).loc[dt]-1
 z=pd.concat([sig.loc[dt],y],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):rows.append((dt,q,len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); x=a.ic
print('assets',len(cl.columns),'dates',len(a),'avgN',a.n.mean(),'coverage',sig.notna().mean().mean(),'IC',x.mean(),'ICIR',x.mean()/x.std(ddof=1),'hit',(x>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
for n in [252,756,1260]:
 q=x.tail(n);print('recent',n,len(q),q.mean(),q.mean()/q.std(ddof=1))
for h in [5,10,20]:
 y=cl.shift(-h)/cl.shift(-1)-1; rr=[]
 for dt in sig.index:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8: rr.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 rr=pd.Series(rr).dropna();print('decay',h,len(rr),rr.mean(),rr.mean()/rr.std(ddof=1))
sig.to_csv('scripts/miner_2_20350820_volume_range_shock_signal.csv')
