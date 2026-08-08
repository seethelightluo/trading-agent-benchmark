"""One idea: rolling return serial-dependence (lag-1 autocorrelation), 60d.
A cross-asset path-structure signal distinct from return level, drawdown and stress-event factors.
Signal is the expanding-safe trailing autocorrelation of daily returns, lagged one day."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change()
# pandas rolling autocorrelation uses only prior/current realized daily returns; signal shifted before forward return matching
F=pd.DataFrame({a:R[a].rolling(60,min_periods=45).corr(R[a].shift(1)) for a in A}).shift(1)
F=F.sub(F.median(axis=1),axis=0)
def met(h,lo=None,hi=None,sgn=1):
 x=(F*sgn).loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index);v=[];nn=[]
 for t in x.index:
  z=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>2:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):v.append(q);nn.append(len(z))
 if not v:return {}
 v=np.array(v);return {'dates':len(v),'ic':round(v.mean(),6),'icir':round(v.mean()/v.std(ddof=1),6),'hit':round((v>0).mean(),4),'mean_n':round(np.mean(nn),2),'min_n':min(nn)}
print('FACTOR return_serial_dependence_60 cutoff',P.index.max().date(),'assets',len(A),'dates',len(P))
print('CELLS',F.notna().sum().sum(),'/',F.size,'coverage',round(F.notna().stack().mean(),6))
for sg,n in [(1,'positive_autocorrelation'),(-1,'negative_autocorrelation')]:
 print('ORIENTATION',n)
 for h in [1,5,10,20]:print('H',h,met(h,sgn=sg))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(P.index.max()-pd.Timedelta(days=180)),None)]:print('REGIME10',n,met(10,lo,hi))
print('TURNOVER',round(F.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CROSS_SECTIONAL_SD',round(F.std(axis=1).mean(),6))
