"""One candidate: peer-relative close-location persistence (20 sessions).
The factor measures whether an asset repeatedly closes near the high or low of
its daily range, median-centers it across the 15 assets, and tests whether this
order-flow/path signal predicts cross-asset forward returns.  The entire
trailing input is shifted one completed session.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date).dt.normalize()
 cols=['open','high','low','close']; d=d[['date']+[c for c in cols if c in d]].copy()
 for c in cols:
  if c not in d:d[c]=np.nan
  d[c]=pd.to_numeric(d[c],errors='coerce')
 return d.groupby('date').last()
D={a:load(a) for a in A}; P=pd.DataFrame({a:D[a]['close'] for a in A}).sort_index();ix=P.index
# Close location value in [-1,1], zero when daily range is absent.
clv=pd.DataFrame({a:((2*D[a]['close']-D[a]['high']-D[a]['low'])/(D[a]['high']-D[a]['low']).replace(0,np.nan)).reindex(ix) for a in A})
raw=clv.rolling(20,min_periods=14).mean()
F=raw.sub(raw.median(axis=1),axis=0).shift(1);cutoff=ix.max()
def met(h,lo=None,hi=None,sgn=1):
 x=(F*sgn).loc[lo:hi];y=(P.shift(-h)/P-1).reindex(x.index); z=[];nn=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);nn.append(len(q))
 if not z:return {'dates':0}
 z=np.asarray(z);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(nn)),2),'min_n':int(min(nn))}
print('FACTOR peer_relative_close_location_persistence_20 cutoff',cutoff.date(),'assets',len(A),'price_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for s,n in [(1,'continuation'),(-1,'inverse')]:
 print('ORIENTATION',n)
 for h in (1,5,10,20):print('H',h,met(h,sgn=s))
print('REGIMES_10')
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cutoff-pd.Timedelta(days=180)),None)]:print(n,'direct',met(10,lo,hi,1),'inverse',met(10,lo,hi,-1))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTION_SD',round(float(F.std(axis=1).mean()),6))
