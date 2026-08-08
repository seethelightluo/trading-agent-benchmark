"""One idea: relative downside-event spacing irregularity (60 sessions).
For each asset's own lagged 60d 20th-percentile loss events, measure the coefficient
of variation of inter-event gaps over a trailing 60d window. This distinguishes
bursty/irregular stress timing from simple event frequency and clustering."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']
event=r.lt(r.rolling(60,min_periods=40).quantile(.20).shift(1))
def irregular(x):
    ids=np.flatnonzero(x.to_numpy(dtype=bool))
    if len(ids)<4:return np.nan
    gaps=np.diff(ids)
    m=gaps.mean()
    return gaps.std(ddof=1)/m if m>0 else np.nan
raw=event.rolling(60,min_periods=30).apply(irregular,raw=False)
# relative across assets, lagged one fully completed session
cand=raw.sub(raw.median(axis=1),axis=0).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]];y=fw[h].reindex(x.index);v=[];b=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8:
   a=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(a):v.append(a);b.append(len(q))
 v=np.array(v)
 return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))} if len(v)>1 else {'dates':len(v)}
print('FACTOR relative_downside_event_spacing_irregularity_60 CUTOFF',cutoff.date(),'ASSETS',len(A),'PERIOD',P.index.min().date(),cutoff.date())
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'STD',round(float(cand.stack().std()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2025_26',('2025-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 print('LIBCORR',n,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
