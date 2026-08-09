import numpy as np,pandas as pd
from scipy.stats import spearmanr
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data,get_index_daily_data
A=get_account_dict()['watch_list']; C={};V={}
for a in A:
 d=get_stock_daily_data(a,5000).copy();d.date=pd.to_datetime(d.date);d=d.sort_values('date').set_index('date');C[a]=pd.to_numeric(d.close,errors='coerce');V[a]=pd.to_numeric(d.volume,errors='coerce').replace(0,np.nan)
P=pd.DataFrame(C);r=P.pct_change();m=r.median(axis=1);cr=r.COPPER
def bt(x,y,w,mn):return x.rolling(w,min_periods=mn).cov(y)/y.rolling(w,min_periods=mn).var()
cand=pd.DataFrame({a:bt(r[a],cr,20,15)-bt(r[a],cr,80,55) for a in A});cand=cand.sub(cand.median(axis=1),axis=0).shift(1)
code=open('scripts/miner_3_20280601_dispersion_shock_beta_resilience_40.py').read();section=code.split('# Full admitted-library signal reconstruction.')[1].split("mx=0;who='';cells=0")[0];exec(section)
mx=0.;who='';cells=0
for n,g in S.items():
 q=pd.concat([cand.stack(),g.stack()],axis=1).replace([np.inf,-np.inf],np.nan).dropna();rho=spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic
 print('LIBCORR',n,'rho',round(float(rho),6),'cells',len(q))
 if abs(rho)>mx:mx=abs(rho);who=n;cells=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(float(mx),6),'FACTOR',who,'EVIDENCE_CELLS',cells,'LIBRARY_FACTORS',len(S))
