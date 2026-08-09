"""One idea: dispersion-regime relative-return asymmetry (60 sessions).
Measures each asset's mean peer-relative return on high cross-asset dispersion days
minus on ordinary dispersion days. It captures a persistent asset-specific response
to fragmented markets, rather than trend, drawdown, or raw macro beta.
"""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff,cs=z['P'],z['S'],z['A'],z['cutoff'],z['cs']
r=P.pct_change(); rel=r.sub(r.median(axis=1),axis=0)
disp=r.std(axis=1); hi=disp>disp.rolling(60,min_periods=40).quantile(.70)
# Both samples must have enough observations; use completed sessions and one-day lag.
hi_mean=rel.where(hi,axis=0).rolling(60,min_periods=12).mean()
lo_mean=rel.where(~hi,axis=0).rolling(60,min_periods=25).mean()
cand=cs(hi_mean-lo_mean).shift(1)
fw={h:P.shift(-h).div(P)-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); vals=[];breadth=[]
 for d in x.index:
  q=pd.concat([x.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):vals.append(v);breadth.append(len(q))
 if not vals:return {'dates':0}
 vals=np.array(vals);return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR dispersion_regime_relative_return_asymmetry_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
