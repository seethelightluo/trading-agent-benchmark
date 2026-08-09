"""Miner 2 single candidate: peer-relative drawdown recovery timing."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d['date']=pd.to_datetime(d.date);C[a]=pd.to_numeric(d.sort_values('date').set_index('date').close,errors='coerce')
P=pd.DataFrame(C).sort_index();R=P.pct_change(fill_method=None); cut=P.dropna(how='all').index.max(); H=[1,5,10,20]
# One idea: an asset that has recovered a greater fraction of its own 60d drawdown over
# the latest 10 sessions than its peers may be in an earlier/later recovery phase.
peak=P.rolling(60,min_periods=40).max(); dd=P/peak-1
old=dd.shift(10); improvement=dd-old
# Scale recovery by drawdown depth at the start, then demean cross-section daily.
F=(improvement/(-old).clip(lower=.002)).clip(-5,5);F=F.sub(F.median(axis=1),axis=0)
fw={h:P.shift(-h)/P-1 for h in H}
def ev(h,span=None):
 x=F if span is None else F.loc[span[0]:span[1]]; y=fw[h].reindex(x.index);z=[];ns=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);ns.append(len(q))
 z=np.array(z)
 return dict(ic_dates=len(z),ic=round(float(z.mean()),6),icir=round(float(z.mean()/z.std(ddof=1)),6),hit_ratio=round(float((z>0).mean()),6),mean_valid_names=round(float(np.mean(ns)),3),min_valid_names=int(min(ns)))
print('FACTOR peer_relative_drawdown_recovery_timing_60_10 CUTOFF',cut.date(),'PERIOD',P.index.min().date(),cut.date(),'ASSETS',len(A))
print('CELLS',int(F.notna().sum().sum()),'/',F.size,'COVERAGE',round(float(F.notna().stack().mean()),6))
for h in H:print('HORIZON',h,ev(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cut.date()))),('recent180',(str(cut-pd.Timedelta(days=180)),str(cut.date())))]:print('REGIME10',n,ev(10,s))
print('TURNOVER_RANK_CHANGE',round(float(F.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
# maintained reconstruction includes all currently admitted signals; absence of evidence fails admission.
code=open('scripts/miner_3_20280601_dispersion_shock_beta_resilience_40.py').read();section=code.split('# Full admitted-library signal reconstruction.')[1].split("mx=0;who='';cells=0")[0];exec(section)
mx=0.;who='';cells=0
for n,g in S.items():
 q=pd.concat([F.stack(),g.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic if len(q)>2 else np.nan
 if np.isfinite(rho) and abs(rho)>mx:mx=float(abs(rho));who=n;cells=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',who,'EVIDENCE_CELLS',cells,'LIBRARY_FACTORS',len(S))
