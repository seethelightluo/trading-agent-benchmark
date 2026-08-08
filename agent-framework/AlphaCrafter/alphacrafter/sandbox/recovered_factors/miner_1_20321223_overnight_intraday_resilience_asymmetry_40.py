"""One idea: peer-relative overnight-to-intraday resilience asymmetry (40 sessions)."""
import runpy,numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_stock_daily_data
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,S,A,cutoff=z['P'],z['S'],z['A'],z['cutoff']
O=pd.DataFrame(index=P.index,columns=A,dtype=float)
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date').loc[lambda q:q.index<=cutoff]
 O[a]=pd.to_numeric(d.open,errors='coerce').reindex(P.index)
# A high score means better relative overnight behavior than relative intraday behavior.
# The continuous difference avoids binary-event ties and is lagged one completed session.
og=O/P.shift(1)-1; ir=P/O-1
rel_og=og.sub(og.median(axis=1),axis=0); rel_ir=ir.sub(ir.median(axis=1),axis=0)
cand=(rel_og-rel_ir).rolling(40,min_periods=24).mean().sub((rel_og-rel_ir).rolling(40,min_periods=24).mean().median(axis=1),axis=0).shift(1)
fw={h:P.pct_change(h).shift(-h) for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); vals=[];bread=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): vals.append(v);bread.append(len(q))
 if len(vals)<2:return {'dates':len(vals)}
 vals=np.array(vals);return {'dates':len(vals),'ic':round(float(vals.mean()),6),'icir':round(float(vals.mean()/vals.std(ddof=1)),6),'hit':round(float((vals>0).mean()),6),'mean_breadth':round(float(np.mean(bread)),3),'min_breadth':int(min(bread))}
print('FACTOR peer_relative_overnight_to_intraday_resilience_asymmetry_40 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20): print('H',h,st(h))
for n,p in [('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1;who='';ev=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna(); rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',ev,'N_FACTORS',len(S))
