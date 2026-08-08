"""One candidate: high-participation peer-relative reversal, available-data validation."""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
A=get_account_dict()['watch_list']; C={}; V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy(); d.date=pd.to_datetime(d.date); d=d.sort_values('date').set_index('date')
 C[a]=pd.to_numeric(d.close,errors='coerce'); V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C); r=P.pct_change(); cutoff=P.dropna(how='all').index.max(); H=[1,5,10,20]
# Negative average peer-relative return only on unusually high own-volume days: transient, well-participated dislocations.
peer=r.sub(r.median(axis=1),axis=0); relv=pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A})
event=relv.gt(relv.rolling(60,min_periods=40).quantile(.65))
f=-peer.where(event).rolling(20,min_periods=8).mean(); f=f.sub(f.median(axis=1),axis=0)
fw={h:P.shift(-h)/P-1 for h in H}
def st(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]]; y=fw[h].reindex(x.index); z=[]; nn=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v): z.append(v);nn.append(len(q))
 z=np.array(z)
 if not len(z): return {'ic_dates':0}
 return {'ic_dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit_ratio':round(float((z>0).mean()),6),'mean_valid_names':round(float(np.mean(nn)),3),'min_valid_names':int(min(nn))}
print('FACTOR high_participation_peer_relative_reversal_20 CUTOFF',cutoff.date(),'INSTRUMENTS',len(A))
print('CELLS',int(f.notna().sum().sum()),'/',f.size,'COVERAGE',round(float(f.notna().stack().mean()),6),'EVENT_RATE',round(float(event.stack().mean()),6))
for h in H: print('HORIZON',h,st(h))
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]: print('REGIME10',n,st(10,s))
print('TURNOVER_RANK_CHANGE',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6))
# Reconstruct admitted library from established comprehensive validator.
code=open('scripts/miner_3_20280601_dispersion_shock_beta_resilience_40.py').read(); section=code.split('# Full admitted-library signal reconstruction.')[1].split("mx=0;who='';cells=0")[0]; exec(section)
mx=0.;who='';ev=0
for n,g in S.items():
 q=pd.concat([f.stack(),g.stack()],axis=1).dropna()
 rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 if abs(rho)>mx: mx=float(abs(rho));who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',who,'EVIDENCE_CELLS',ev,'LIBRARY_FACTORS',len(S))
