"""One idea: peer-relative overnight-gap recovery resilience (60 days).
Scores an asset by its average intraday recovery following its own negative
opening gaps, minus the contemporaneous peer median.  This separates how an
asset absorbs overnight information from ordinary close-to-close reversal.
"""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff,cs=z['P'],z['r'],z['S'],z['A'],z['cutoff'],z['cs']
# Load visible OHLC independently; all fields are used at signal close then lagged.
from alphacrafter.sim.utils import get_stock_daily_data
frames={}
for a in A:
    d=get_stock_daily_data(a).copy(); d.index=pd.to_datetime(d.index)
    frames[a]=d
O=pd.DataFrame({a:frames[a]['open'] for a in A}).reindex(P.index)
C=pd.DataFrame({a:frames[a]['close'] for a in A}).reindex(P.index)
gap=O.div(C.shift(1)).sub(1)
intra=C.div(O).sub(1)
# Normalize rebound by initial gap magnitude; require a meaningful negative gap.
event=gap.lt(-.003)
recovery=(intra.div((-gap).clip(lower=.003))).where(event)
raw=recovery.rolling(60,min_periods=12).mean()
peer=raw.apply(lambda x:x.sub(x.median()),axis=1)
cand=cs(peer).shift(1)
fw={h:P.shift(-h).div(P)-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); vs=[];bs=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):vs.append(v);bs.append(len(q))
 if not vs:return {'dates':0}
 v=np.array(vs);return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(bs)),3),'min_breadth':int(min(bs))}
print('FACTOR peer_relative_overnight_gap_recovery_resilience_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6),'EVENT_RATE',round(float(event.stack().mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
