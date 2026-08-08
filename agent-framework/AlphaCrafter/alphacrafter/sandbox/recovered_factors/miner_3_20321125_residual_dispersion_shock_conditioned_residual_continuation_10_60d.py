"""One candidate: residual-dispersion-shock conditioned residual continuation.
On each completed date, score each asset by its 10d market-beta-residual return,
scaled by trailing 20d residual volatility.  The signal is active only after a
cross-sectional residual-dispersion expansion: current 10d mean dispersion is
above its preceding 10d mean. Forward returns are research labels only and all
inputs end at CUT.
"""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2032-11-24')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'].rename(a)
p=pd.concat([load(a) for a in A],axis=1).sort_index().loc[:CUT]
r=p.pct_change(); m=r.mean(axis=1)
beta=r.apply(lambda x:x.rolling(60,min_periods=42).cov(m)).div(m.rolling(60,min_periods=42).var()+1e-12,axis=0)
e=r-beta.mul(m,axis=0)
# residual continuation normalized by its own recent idiosyncratic risk
base=e.rolling(10,min_periods=8).sum().div(e.rolling(20,min_periods=14).std()+1e-12)
# dispersion shock uses only contemporaneous/past residual observations
csdisp=e.std(axis=1,ddof=1)
recent=csdisp.rolling(10,min_periods=8).mean(); prior=recent.shift(10)
state=(recent/prior-1)>0
f=base.where(state, np.nan)
print('CANDIDATE residual_dispersion_shock_conditioned_residual_continuation_10_60d cutoff',CUT.date(),'calendar_dates',len(p),'assets',len(A))
print('active_dates',int(f.notna().any(axis=1).sum()),'coverage',round(float(f.notna().mean().mean()),6),'valid_cells',int(f.notna().sum().sum()),'dispersion_expansion_frequency',round(float(state.mean()),4))
ics={}
for h in (1,3,5,7,10,20):
 fw=p.shift(-h).div(p)-1; vals=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z)); ns.append(len(q))
 x=pd.Series(dict(vals),dtype=float); ics[h]=x; sd=x.std(ddof=1)
 print('H%d IC=%.6f ICIR=%.6f dates=%d hit=%.4f meanN=%.2f'%(h,x.mean(),x.mean()/sd,len(x),(x>0).mean(),np.mean(ns)))
 if h==10:
  for n,lo,hi in [('2020-2024','2020-01-01','2024-12-31'),('2025-2026','2025-01-01','2026-12-31'),('2027+','2027-01-01',str(CUT.date()))]:
   z=x.loc[lo:hi]; print('REGIME10',n,'dates',len(z),'IC',round(z.mean(),6),'ICIR',round(z.mean()/z.std(ddof=1),6),'hit',round((z>0).mean(),4))
rk=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('RANK_TURNOVER',round(float(np.mean(turn)),6),'pairs',len(turn))
print('DECAY',{h:(round(float(x.mean()),6),round(float(x.mean()/x.std(ddof=1)),6),len(x)) for h,x in ics.items()})
