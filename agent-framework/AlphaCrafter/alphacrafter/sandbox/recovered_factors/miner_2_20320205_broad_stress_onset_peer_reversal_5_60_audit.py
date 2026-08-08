"""Full admitted-library novelty audit for broad-stress-onset peer reversal."""
import runpy,numpy as np
from scipy.stats import spearmanr
z=runpy.run_path('scripts/miner_2_20311211_dxy_downside_lag5_full_library_reaudit.py')
P,r,S=z['P'],z['r'],z['S']; ret5=P.pct_change(5); broad=ret5.median(axis=1)
gate=(broad<=broad.rolling(60,min_periods=45).quantile(.20)).astype(float)
cand=(-ret5.sub(ret5.median(axis=1),axis=0)).mul(gate,axis=0).shift(1)
mx=0;who=''; evidence=0
for name,s in S.items():
 q=cand.stack().to_frame('c').join(s.stack().to_frame('s'),how='inner').dropna()
 rho=spearmanr(q.c,q.s).statistic if len(q) else np.nan
 print('LIBCORR',name,'cells',len(q),'rho',round(float(rho),6) if np.isfinite(rho) else 'INVALID')
 if np.isfinite(rho) and abs(rho)>mx: mx=abs(rho);who=name;evidence=len(q)
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'MOST',who,'EVIDENCE',evidence,'N_FACTORS',len(S))
