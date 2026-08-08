"""One idea: continuous overnight-to-intraday absorption resilience (60 sessions).
Measures an asset's volatility-normalized tendency to offset (rather than extend)
its overnight move, relative to contemporaneous cross-asset peers.
"""
import runpy, numpy as np, pandas as pd
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S,A,cutoff,cs=z['P'],z['r'],z['S'],z['A'],z['cutoff'],z['cs']
from alphacrafter.sim.utils import get_stock_daily_data
F={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d['date']=pd.to_datetime(d.date)
 F[a]=d.sort_values('date').set_index('date')
O=pd.DataFrame({a:pd.to_numeric(F[a].open,errors='coerce') for a in A}).reindex(P.index)
C=pd.DataFrame({a:pd.to_numeric(F[a].close,errors='coerce') for a in A}).reindex(P.index)
gap=O.div(C.shift(1)).sub(1); intra=C.div(O).sub(1)
# positive if intraday price action offsets the signed overnight move; scale by own recent total-return volatility
vol=r.rolling(20,min_periods=12).std().clip(lower=1e-5)
daily=(-gap*intra).div(vol**2)
raw=daily.rolling(60,min_periods=35).mean()
cand=cs(raw.sub(raw.median(axis=1),axis=0)).shift(1)
fw={h:P.shift(-h).div(P)-1 for h in (1,5,10,20)}
def st(h,period=None):
 x=cand if period is None else cand.loc[period[0]:period[1]]; y=fw[h].reindex(x.index); v=[]; b=[]
 for dt in x.index:
  q=pd.concat([x.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1:
   u=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(u):v.append(u);b.append(len(q))
 if not v:return {'dates':0}
 v=np.array(v); return {'dates':len(v),'ic':round(float(v.mean()),6),'icir':round(float(v.mean()/v.std(ddof=1)),6),'hit':round(float((v>0).mean()),6),'mean_breadth':round(float(np.mean(b)),3),'min_breadth':int(min(b))}
print('FACTOR continuous_overnight_intraday_absorption_resilience_60 CUTOFF',cutoff.date(),'ASSETS',len(A))
print('CELLS',int(cand.notna().sum().sum()),'/',cand.size,'COVERAGE',round(float(cand.notna().stack().mean()),6),'TURNOVER',round(float(cand.rank(axis=1,pct=True).diff().abs().stack().mean()),6),'CS_STD',round(float(cand.std(axis=1).mean()),6))
for h in (1,5,10,20):print('H',h,st(h))
for n,p in [('2020_22',('2020-01-01','2022-12-31')),('2023_26',('2023-01-01','2026-12-31')),('2027_now',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,p))
mx=-1; who=''; evidence=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>=8 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=abs(rho);who=n;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S))
