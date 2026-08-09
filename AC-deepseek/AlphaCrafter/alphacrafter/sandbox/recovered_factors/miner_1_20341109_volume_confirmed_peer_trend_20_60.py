"""One candidate: peer-relative volume-confirmed medium-term trend (20 sessions).
Signal is 20-day return multiplied by each asset's 20-day volume participation relative
to its own 60-day median, then cross-sectionally standardized. It tests whether price
trends backed by unusually sustained tradable participation persist across assets.
"""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff,cs=z['P'],z['S'],z['A'],z['cutoff'],z['cs']
# Extract volume from the audit loader if supplied; otherwise obtain visible stock data is not
# appropriate for historical audit, so fail explicitly rather than fabricate evidence.
print('LOADER_KEYS', sorted(z.keys()))
V=z.get('V')
if V is None:
    raise RuntimeError('Shared audit loader did not expose volume matrix')
r20=P/P.shift(20)-1
part=V.rolling(20,min_periods=15).mean()/V.rolling(60,min_periods=40).median()
raw=r20*part
cand=cs(raw).shift(1); fw={h:P.shift(-h)/P-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; vals=[]; breadth=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],fw[h].loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v); breadth.append(len(q))
 if not vals:return {'dates':0}
 v=np.array(vals); return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(breadth)),3),'min_breadth':int(min(breadth))}
print('FACTOR volume_confirmed_peer_trend_20_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20): print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx: mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
