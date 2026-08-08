"""One idea: residual five-session reversal conditioned on broad downside breadth."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
CUTOFF=pd.Timestamp('2034-05-24')
def close(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:close(a) for a in A}).loc[:CUTOFF]; cut=p.dropna(how='all').index.max();p=p.loc[:cut]
r=p.pct_change(); r5=p.pct_change(5); m=r.mean(axis=1); v=m.rolling(60,min_periods=40).var()
b=pd.DataFrame({a:r[a].rolling(60,min_periods=40).cov(m).div(v) for a in A})
# Reversal only when at least 60% of the universe has lost value over five days.
# This distinguishes systemic downside washouts from ordinary short-horizon noise.
breadth=(r5<0).mean(axis=1); state=breadth>=.60
f=(-(r-b.mul(m,axis=0)).rolling(5,min_periods=5).sum()).where(state,0.).replace([np.inf,-np.inf],np.nan)
print('FACTOR residual_broad_downside_gated_short_reversal_5_60d VALIDATED_THROUGH',cut.date())
print('definition=-sum_5(residual daily return versus 60d equal-weight market beta), activated only when >=60% assets have negative trailing-5d returns')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f active_dates=%d active_rate=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),state.sum(),state.mean()))
ics={}
for h in [1,5,10,20]:
 fut=p.shift(-h).div(p)-1; out=[];ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fut.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z):out.append((d,z));ns.append(len(q))
 s=pd.Series(dict(out),dtype=float);ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.6f meanN=%.4f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for nm,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2034','2027-01-01',cut)]:
 s=ics[5].loc[lo:hi]; print('REGIME5 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.6f'%(nm,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
ranks=f.rank(axis=1,pct=True);turn=[]
for i in range(1,len(ranks)):
 q=ranks.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:turn.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turn),len(turn)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  z=json.load(open(fn))
  if z.get('validation',{}).get('status')=='EFFECTIVE':eff.append(z['factor_id'])
 except Exception:pass
scores=[];missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits:missing.append(fid);continue
 old=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),old.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
if len(scores)==len(eff):print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else:print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
f.to_pickle('scripts/miner_3_20340525_residual_broad_downside_gated_short_reversal_5_60d_signal.pkl')
