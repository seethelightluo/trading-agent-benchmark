"""One idea: trend-orthogonal mild-broad-pullback relative capture (60 sessions).
Measure an asset's average contemporaneous return minus cross-asset median only on
ordinary broad pullback days (equal-weight return below zero but above its trailing
20th-percentile stress tail). Remove its current 20-session trend cross-sectionally,
so this asks whether non-tail dip resilience distinct from directional momentum
predicts future relative returns. All inputs are lagged by one completed session.
"""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff=z['P'],z['r'],z['S'],z['A'],z['cutoff']; cs=z['cs']
broad=r.mean(axis=1); tail=broad.rolling(60,min_periods=40).quantile(.20)
mild=(broad<0)&(broad>=tail)
relative=r.sub(r.median(axis=1),axis=0)
raw=relative.where(mild,axis=0).rolling(60,min_periods=20).mean()
trend=P/P.shift(20)-1
# daily cross-sectional residual from raw ~= alpha + beta*trend
cand=pd.DataFrame(index=P.index,columns=A,dtype=float)
for dt in P.index:
 q=pd.concat([raw.loc[dt],trend.loc[dt]],axis=1).dropna()
 if len(q)>=8 and q.iloc[:,1].std()>0:
  beta=np.cov(q.iloc[:,0],q.iloc[:,1],ddof=1)[0,1]/q.iloc[:,1].var(ddof=1)
  cand.loc[dt]=raw.loc[dt]-beta*trend.loc[dt]
cand=cs(cand).shift(1)
fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def stats(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); v=[]; b=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   a=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(a):v.append(a);b.append(len(q))
 v=np.array(v)
 if len(v)<2:return {'dates':len(v)}
 return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR trend_orthogonal_mild_broad_pullback_relative_capture_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6),'MILD_FRACTION',round(float(mild.mean()),6))
for h in (1,5,10,20):print('H',h,stats(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,stats(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
