"""One idea: peer-relative trend conditioned by asset-specific volatility contraction (20/10/60)."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C); R=P.pct_change()
def cs(x): return x.sub(x.median(axis=1),axis=0)
# A positive score is a relatively strong 20-session performer whose last 10-session
# realized volatility is low versus its own 60-session baseline: a quiet, persistent trend.
r20=P.pct_change(20)
v10=R.rolling(10,min_periods=8).std(); v60=R.rolling(60,min_periods=45).std()
raw=cs(r20)*(v60/v10).clip(.25,4)
sig=cs(raw).clip(-1,1).shift(1)
fwd={h:P.shift(-h).div(P).sub(1) for h in (1,5,10,20)}
def stat(h,lo=None,hi=None):
 x=sig.loc[lo:hi] if lo else sig; z=[]; br=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fwd[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(rho):z.append(rho);br.append(len(q))
 z=np.asarray(z); return {'dates':len(z),'ic':round(z.mean(),6),'icir':round(z.mean()/z.std(ddof=1),6),'hit':round((z>0).mean(),6),'breadth':round(np.mean(br),3),'min_breadth':min(br)}
cut=P.dropna(how='all').index.max(); rank=sig.rank(axis=1,pct=True)
print('FACTOR quiet_trend_volatility_contraction_20_10_60 CUTOFF',cut.date(),'ASSETS',len(A))
print('CELLS',int(sig.notna().sum().sum()),'/',sig.size,'COVERAGE',round(sig.notna().stack().mean(),6),'TURNOVER',round(rank.diff().abs().stack().mean(),6))
for h in (1,5,10,20):print('H',h,stat(h))
for n,lo,hi in [('2025_26','2025-01-01','2026-12-31'),('2027_now','2027-01-01',str(cut.date())),('recent180',str(cut-pd.Timedelta(days=180)),str(cut.date()))]: print('REGIME10',n,stat(10,lo,hi))
for n,x in {'risk_adjusted_trend20':cs(r20/v20) if (v20:=R.rolling(20,min_periods=15).std()) is not None else r20,'quiet_path_proxy':cs(r20)*(1/(R.rolling(20,min_periods=15).std())),'vol_contraction':cs(v60/v10)}.items():
 q=pd.concat([sig.stack(),x.shift(1).stack()],axis=1).dropna();print('PROXY',n,'cells',len(q),'rho',round(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic,6))
