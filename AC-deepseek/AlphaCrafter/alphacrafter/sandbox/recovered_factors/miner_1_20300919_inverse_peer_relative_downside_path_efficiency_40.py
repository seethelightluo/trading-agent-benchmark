"""One candidate: inverse peer-relative downside-path efficiency, 40 sessions.
Tests whether jagged rather than smooth relative performance on broad down days
predicts cross-asset rebound/convexity. This is a separately oriented factor.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict, get_stock_daily_data
A=get_account_dict()['watch_list']
def load(a):
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date).dt.normalize()
 return pd.Series(pd.to_numeric(d.close,errors='coerce').to_numpy(),index=d.date).groupby(level=0).last()
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); med=R.median(1); rel=R.sub(med,axis=0)
# Negative path efficiency: large absolute relative fluctuations with little/net negative
# cumulative resilience on broad down sessions receives a high score.
down=rel.where(med.lt(0),axis=0); net=down.rolling(40,min_periods=12).sum(); path=down.abs().rolling(40,min_periods=12).sum()
F=(-(net/path.replace(0,np.nan))).sub((-(net/path.replace(0,np.nan))).median(1),axis=0).shift(1)
def met(h,lo=None,hi=None):
 x=F.loc[lo:hi]; y=P.shift(-h).div(P).sub(1).reindex(x.index); z=[]; ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>2:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v); ns.append(len(q))
 z=np.array(z)
 return {'dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit':round(float((z>0).mean()),4),'mean_n':round(float(np.mean(ns)),2),'min_n':int(min(ns))} if len(z) else {'dates':0}
cut=P.index.max(); print('FACTOR inverse_peer_relative_downside_path_efficiency_40 CUTOFF',cut.date(),'ASSETS',len(A),'PRICE_DATES',len(P))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'COVERAGE',round(float(F.notna().stack().mean()),6),'TURNOVER',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(F.std(1).mean()),6))
for h in [1,5,10,20]: print('H',h,met(h))
for n,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029_current','2029-01-01',None),('recent180',str(cut-pd.Timedelta(days=180)),None)]: print('REGIME10',n,met(10,lo,hi))
print('NOVELTY_AUDIT failed: no reproducible aligned daily histories for every admitted library signal; mandatory max_abs_library_correlation cannot be evidenced.')
