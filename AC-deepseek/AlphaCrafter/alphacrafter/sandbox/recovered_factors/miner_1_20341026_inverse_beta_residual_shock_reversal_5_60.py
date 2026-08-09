"""One idea: inverse idiosyncratic shock after systematic cross-asset move.
Scores assets that underperformed their own 60d beta-implied move over the last
5 sessions. This is an interpretable residual-return reversal, distinct from
absolute drawdown and ordinary peer-relative return by estimating each asset's
systematic sensitivity first.
"""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff,cs=z['P'],z['S'],z['A'],z['cutoff'],z['cs']
r1=P.pct_change(); m=r1.median(axis=1)
# trailing (completed) beta of each asset to robust cross-asset daily move
beta=r1.rolling(60,min_periods=40).cov(m).div(m.rolling(60,min_periods=40).var().replace(0,np.nan),axis=0).shift(1)
r5=P.pct_change(5)
market5=m.rolling(5).sum()
resid=r5-beta.mul(market5,axis=0)
raw=-resid # negative idiosyncratic shock is expected to mean revert
cand=cs(raw).shift(1)
fw={h:P.shift(-h).div(P)-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); v=[]; b=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   a=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(a):v.append(a);b.append(len(q))
 if not v:return {'dates':0}
 v=np.array(v);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR inverse_beta_residual_shock_reversal_5_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
