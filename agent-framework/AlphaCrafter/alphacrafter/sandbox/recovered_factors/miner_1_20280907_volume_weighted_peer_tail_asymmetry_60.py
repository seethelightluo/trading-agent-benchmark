"""One candidate: volume-weighted peer-relative tail asymmetry, 60 observations."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list'];C={};V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date')
 C[a]=pd.to_numeric(d.close,errors='coerce');V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C);r=P.pct_change();peer=r.sub(r.median(axis=1),axis=0);cutoff=P.dropna(how='all').index.max();H=[1,5,10,20]
# Continuous own-volume surprise weights distinguish constructive from destructive peer-relative tails.
relv=pd.DataFrame({a:np.log(V[a]/V[a].rolling(20,min_periods=15).mean()) for a in A})
part=relv.rolling(60,min_periods=40).rank(pct=True)
pos=(peer.clip(lower=0)*part).rolling(60,min_periods=40).sum();neg=((-peer.clip(upper=0))*part).rolling(60,min_periods=40).sum()
f=(pos-neg)/(pos+neg).replace(0,np.nan);f=f.sub(f.median(axis=1),axis=0);fw={h:P.shift(-h)/P-1 for h in H}
def st(h,span=None):
 x=f if span is None else f.loc[span[0]:span[1]];y=fw[h].reindex(x.index);z=[];nn=[]
 for t in x.index:
  q=pd.concat([x.loc[t],y.loc[t]],axis=1).dropna()
  if len(q)>=8:
   v=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
   if np.isfinite(v):z.append(v);nn.append(len(q))
 z=np.array(z)
 return {'ic_dates':len(z),'ic':round(float(z.mean()),6),'icir':round(float(z.mean()/z.std(ddof=1)),6),'hit_ratio':round(float((z>0).mean()),6),'mean_valid_names':round(float(np.mean(nn)),3),'min_valid_names':int(min(nn))} if len(z) else {'ic_dates':0}
print('FACTOR volume_weighted_peer_tail_asymmetry_60 CUTOFF',cutoff.date(),'INSTRUMENTS',len(A),flush=True);print('CELLS',int(f.notna().sum().sum()),'/',f.size,'COVERAGE',round(float(f.notna().stack().mean()),6),flush=True)
for h in H:print('HORIZON',h,st(h),flush=True)
for n,s in [('2020',('2020-01-01','2020-12-31')),('2021_22',('2021-01-01','2022-12-31')),('2023_24',('2023-01-01','2024-12-31')),('2025_26',('2025-01-01','2026-12-31')),('2027_current',('2027-01-01',str(cutoff.date()))),('recent180',(str(cutoff-pd.Timedelta(days=180)),str(cutoff.date())))]:print('REGIME10',n,st(10,s),flush=True)
print('TURNOVER_RANK_CHANGE',round(float(f.rank(axis=1,pct=True).diff().abs().stack().mean()),6),flush=True)
code=open('scripts/miner_3_20280601_dispersion_shock_beta_resilience_40.py').read();section=code.split('# Full admitted-library signal reconstruction.')[1].split("mx=0;who='';cells=0")[0];exec(section)
mx=0.;who='';ev=0
for n,g in S.items():
 q=pd.concat([f.stack(),g.stack()],axis=1).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 if abs(rho)>mx:mx=float(abs(rho));who=n;ev=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',who,'EVIDENCE_CELLS',ev,'LIBRARY_FACTORS',len(S))
