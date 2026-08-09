"""One candidate: dispersion-conditioned five-day cross-sectional reversal; validation through prior completed bar."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-12-21')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT,'close'] for a in A})
r=p.pct_change(); disp=r.std(axis=1,ddof=0)
state=disp>disp.rolling(60,min_periods=40).quantile(.75)
f=(-p.pct_change(5)).where(state,0.).replace([np.inf,-np.inf],np.nan)
print('FACTOR dispersion_conditioned_short_reversal_5d VALIDATED_THROUGH',CUT.date())
print('cells=%d dates=%d coverage=%.6f activation=%.6f assets=%d'%(f.notna().sum().sum(),f.notna().any(axis=1).sum(),f.notna().mean().mean(),state.mean(),len(A)))
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1; out=[]; nn=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=spearmanr(q.f,q.y).statistic
   if np.isfinite(v):out.append((d,v));nn.append(len(q))
 s=pd.Series(dict(out)); print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(nn)))
 if h==5:
  for name,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01',CUT)]:
   x=s.loc[lo:hi];print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(name,len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean()))
rk=f.rank(axis=1,pct=True); tt=[]
for j in range(1,len(rk)):
 q=rk.iloc[[j-1,j]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:tt.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(tt),len(tt)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn))
  if z.get('validation',{}).get('status')=='EFFECTIVE':eff.append(z['factor_id'])
 except:pass
scores=[]; missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits: missing.append(fid);continue
 z=pd.read_pickle(max(hits,key=os.path.getmtime))
 q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
print('MISSING',missing)
if len(scores)==len(eff):print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else: print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
f.to_pickle('scripts/miner_3_20331222_dispersion_conditioned_short_reversal_5d_signal.pkl')
