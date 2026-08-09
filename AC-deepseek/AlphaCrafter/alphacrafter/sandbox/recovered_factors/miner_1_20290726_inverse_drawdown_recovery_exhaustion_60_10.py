"""One idea: inverse drawdown-recovery convexity (60/10).
Explicit mean-reversion hypothesis: a large rebound following a deep, established
drawdown is exhaustion rather than recovery continuation; score favours smaller or
negative rebounds conditional on prior drawdown depth.  The sign is specified before
validation, separately from the prior positive continuation candidate.
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date)
 C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C).sort_index(); peak=P.rolling(60,min_periods=45).max()
# Peak/drawdown is frozen immediately before the latest 10-session rebound window.
prior_dd=(P.shift(10)/peak.shift(10)-1); rebound=P/P.shift(10)-1
f=(-(rebound*(-prior_dd))).shift(1); f=f.sub(f.median(axis=1),axis=0)
H=(1,5,10,20); FW={h:P.shift(-h).div(P)-1 for h in H};cutoff=P.dropna(how='all').index.max()
def ev(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]];y=FW[h].reindex(x.index);v=[];n=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(z):v.append(z);n.append(len(q))
 if not v:return {'dates':0}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),4),'mean_n':round(float(np.mean(n)),2),'min_n':int(min(n))}
print('FACTOR inverse_drawdown_recovery_exhaustion_60_10 cutoff',cutoff.date(),'assets',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'coverage',round(float(f.notna().stack().mean()),6),'mean_names',round(float(f.notna().sum(axis=1).mean()),2))
print('MEAN_PRIOR_DD',round(float(prior_dd.mean().mean()),6),'MEAN_REBOUND',round(float(rebound.mean().mean()),6))
for h in H:print('H',h,ev(h))
for name,span in [('2020_22',('2020-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',name,ev(10,span))
print('TURNOVER',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CROSS_SECTIONAL_SD',round(float(f.std(axis=1).mean()),6))
