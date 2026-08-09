"""One idea: beta-neutral residual trend acceleration (10/40).
For each asset, remove its rolling 40-session leave-one-out peer-median beta.
The signal is recent 10-session residual return minus prior 30-session residual
return, scaled by 40-session residual volatility and median-centered. It seeks
idiosyncratic trend acceleration independent of broad cross-asset direction.
All signals are lagged one completed session."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C).sort_index(); R=P.pct_change(); cutoff=P.dropna(how='all').index.max()
res=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 peer=R.drop(columns=a).median(axis=1)
 beta=R[a].rolling(40,min_periods=30).cov(peer)/peer.rolling(40,min_periods=30).var()
 res[a]=R[a]-beta*peer
# recent residual momentum versus the preceding 30 sessions; normalize only scale, not direction.
r10=res.rolling(10,min_periods=8).sum(); prior30=res.shift(10).rolling(30,min_periods=23).sum()
rv=res.rolling(40,min_periods=30).std()*np.sqrt(40)
raw=(r10-prior30)/rv
cand=raw.sub(raw.median(axis=1),axis=0).clip(-5,5).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stats(h,lo=None,hi=None):
 x=cand.loc[lo:hi] if lo else cand; z=[]; b=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);b.append(len(q))
 z=np.array(z)
 return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR beta_neutral_residual_trend_acceleration_10_40 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for n,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cutoff.date())),('recent180',str(cutoff-pd.Timedelta(days=180)),str(cutoff.date()))]:print('REGIME10',n,stats(10,lo,hi))
# proximal diagnostic, not an admission-quality library audit
rel10=P.pct_change(10).sub(P.pct_change(10).median(axis=1),axis=0)
q=pd.concat([cand.stack(),rel10.stack()],axis=1).dropna()
print('PROXY_CORR relative_return_10 cells',len(q),'rho',round(float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic),6))
print('ADMISSION_NOTE exact correlation against every effective library factor is required before persistence.')
