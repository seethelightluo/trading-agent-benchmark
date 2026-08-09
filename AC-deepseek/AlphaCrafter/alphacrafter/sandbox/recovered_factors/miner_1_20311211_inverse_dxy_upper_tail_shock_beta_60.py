"""One idea: inverse DXY upper-tail shock beta (60d), cross-asset resilience."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C); R=P.pct_change(); cutoff=P.dropna(how='all').index.max()
d=get_index_daily_data('DXY',5000).copy();d.date=pd.to_datetime(d.date)
dx=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce').reindex(P.index).pct_change()
# On DXY's own high positive-shock sessions (ex ante 80th-percentile 60d), measure each asset's peer-relative response; invert beta.
th=dx.rolling(60,min_periods=40).quantile(.80).shift(1); shock=dx.gt(th)
rel=R.sub(R.median(axis=1),axis=0); x=dx.where(shock)
# slope through origin using only shock observations, no same-day signal use due lag
f=-rel.mul(x,axis=0).rolling(60,min_periods=12).sum().div(x.pow(2).rolling(60,min_periods=12).sum(),axis=0)
f=f.sub(f.median(axis=1),axis=0).shift(1); fw={h:P.shift(-h).div(P)-1 for h in [1,5,10,20]}
def stat(h,sl=None):
 qf=f if sl is None else f.loc[sl[0]:sl[1]]; z=[]; ns=[]
 for t in qf.index:
  q=pd.concat([qf.loc[t],fw[h].loc[t]],axis=1).dropna()
  if len(q)>=8: z.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
 if not z:return {}
 z=np.array(z);return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),6),'breadth':round(float(np.mean(ns)),3),'min_breadth':int(min(ns))}
print('FACTOR inverse_dxy_upper_tail_shock_beta_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'COVERAGE',round(float(f.notna().stack().mean()),6),'TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'DISPERSION',round(float(f.std(axis=1).mean()),6),'SHOCK_DAYS',int(shock.sum()))
for h in fw:print('H',h,stat(h))
for n,s in [('2025_2026',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stat(10,s))
