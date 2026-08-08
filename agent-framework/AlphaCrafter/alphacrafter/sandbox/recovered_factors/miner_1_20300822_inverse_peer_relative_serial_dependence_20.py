"""One candidate: peer-relative return serial-dependence reversal, 20 sessions."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date).dt.normalize();return pd.Series(pd.to_numeric(d.close,errors='coerce').values,index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index();R=P.pct_change();peer=R.median(axis=1);rel=R.sub(peer,axis=0);cut=P.index.max()
# Lag-1 autocorrelation of peer-relative daily returns. Persistent positive serial dependence is scored negatively (mean-reversion preference).
def ac(x): return x.rolling(20,min_periods=16).corr(x.shift(1))
F=(-pd.DataFrame({a:ac(rel[a]) for a in A}));F=F.sub(F.median(axis=1),axis=0).shift(1)
def met(h,lo=None,hi=None,sgn=1):
 x=(F*sgn).loc[lo:hi];y=(P.shift(-h)/P-1).reindex(x.index);z=[];nn=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);nn.append(len(q))
 if not z:return {}
 z=np.array(z);return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),4),'mean_n':round(np.mean(nn),2),'min_n':min(nn)}
print('FACTOR inverse_peer_relative_serial_dependence_20 CUTOFF',cut.date(),'ASSETS',len(A),'DATES',len(P))
print('CELLS',F.notna().sum().sum(),'/',F.size,'COVERAGE',round(F.notna().stack().mean(),6),'TURNOVER',round(F.rank(axis=1,pct=True).diff().abs().stack().mean(),6),'CS_STD',round(F.std(axis=1).mean(),6))
for s,n in [(1,'inverse_autocorr'),(-1,'autocorr')]:
 print('ORIENT',n)
 for h in (1,5,10,20):print('H',h,met(h,sgn=s))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]:print('REGIME10',n,met(10,lo,hi))
