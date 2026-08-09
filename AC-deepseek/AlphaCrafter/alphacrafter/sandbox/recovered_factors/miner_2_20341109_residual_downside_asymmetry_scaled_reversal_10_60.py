"""One candidate: residual downside-asymmetry conditioned 10-session reversal."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def close(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()['close']
p=pd.DataFrame({a:close(a) for a in A})
CUT=min(p.dropna(how='all').index.max(),pd.Timestamp('2034-11-08'));p=p.loc[:CUT];r=p.pct_change()
# All components are lagged: asset residual returns versus equal-weight market; scale reversal
# only when its 60d downside semivolatility dominates upside semivolatility.
m=r.mean(axis=1);res=r.sub(m,axis=0)
down=np.sqrt(res.where(res<0,0).pow(2).rolling(60,min_periods=45).mean()).shift(1)
up=np.sqrt(res.where(res>0,0).pow(2).rolling(60,min_periods=45).mean()).shift(1)
asym=(down/(up+1e-12)).clip(0.25,4)
base=-res.rolling(10,min_periods=8).sum().shift(1)
f=base*asym
print('FACTOR residual_downside_asymmetry_scaled_reversal_10_60 VALIDATED_THROUGH',CUT.date())
print('definition=negative completed 10-session asset return residual to equal-weight cross-asset return, multiplied by lagged 60-session residual downside semivolatility divided by upside semivolatility; ratio capped [0.25,4]')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1;obs=[];ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   c=spearmanr(q.f,q.y).statistic
   if np.isfinite(c):obs.append((d,c));ns.append(len(q))
 s=pd.Series(dict(obs),dtype=float);ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_33','2027-01-01','2033-12-31'),('2034_YTD','2034-01-01',CUT)]:
 s=ics[10].loc[lo:hi];print('REGIME10 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(n,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True);to=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:to.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(to),len(to)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  j=json.load(open(fn))
  if j.get('validation',{}).get('status')=='EFFECTIVE':eff.append(j['factor_id'])
 except Exception:pass
sc=[];missing=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if not hits:missing.append(fid);continue
 z=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 sc.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(sc),len(missing)))
print('MAX_ABS_LIBRARY_CORRELATION='+('%.6f'%max(sc)[0] if len(sc)==len(eff) else 'UNAVAILABLE'))
f.to_pickle('scripts/miner_2_20341109_residual_downside_asymmetry_scaled_reversal_10_60_signal.pkl')
