"""One candidate: peer-beta-adjusted residual momentum (20d signal, 60d beta).
An asset's recent performance is separated from its contemporaneous broad
cross-asset move, then standardized by its residual volatility.  The lagged
signal tests whether idiosyncratic leadership persists rather than merely
loading on a common risk-on/risk-off shock.
"""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 d['close']=pd.to_numeric(d['close'],errors='coerce')
 return d[['date','close']].groupby('date').last()['close']
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(fill_method=None)
# Leave-one-out median market return prevents an asset feeding its own benchmark.
M=pd.DataFrame({a:R.drop(columns=a).median(axis=1) for a in A})
res=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 cov=R[a].rolling(60,min_periods=40).cov(M[a]); var=M[a].rolling(60,min_periods=40).var().replace(0,np.nan)
 res[a]=R[a]-(cov/var)*M[a]
raw=res.rolling(20,min_periods=14).sum()/res.rolling(60,min_periods=40).std().replace(0,np.nan)
F=raw.sub(raw.median(axis=1),axis=0).shift(1); cutoff=P.index.max()
def met(h,lo=None,hi=None,sgn=1):
 x=(F*sgn).loc[lo:hi]; y=(P.shift(-h)/P-1).reindex(x.index); z=[]; nn=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);nn.append(len(q))
 if not z:return {'dates':0}
 z=np.array(z); return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(nn)),2),'min_n':int(min(nn))}
print('FACTOR peer_beta_adjusted_residual_momentum_20_60 cutoff',cutoff.date(),'assets',len(A),'price_dates',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'coverage',round(float(F.notna().stack().mean()),6))
for s,n in [(1,'continuation'),(-1,'inverse')]:
 print('ORIENTATION',n)
 for h in (1,5,10,20): print('H',h,met(h,sgn=s))
print('REGIMES_10')
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cutoff-pd.Timedelta(days=180)),None)]:
 print(n,'direct',met(10,lo,hi,1),'inverse',met(10,lo,hi,-1))
print('TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTION_SD',round(float(F.std(axis=1).mean()),6))
