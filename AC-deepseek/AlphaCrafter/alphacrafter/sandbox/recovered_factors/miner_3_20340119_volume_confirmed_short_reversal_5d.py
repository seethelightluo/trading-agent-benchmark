"""One candidate: volume-confirmed short reversal, evaluated only through 2034-01-18."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2034-01-18')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def series(a,col):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index()
 return d.loc[:CUT,col] if col in d else pd.Series(dtype=float)
p=pd.DataFrame({a:series(a,'close') for a in A})
v=pd.DataFrame({a:series(a,'volume') for a in A}).reindex(p.index)
# A short move accompanied by unusually heavy own trading is interpreted as a transient
# liquidity/positioning shock; score is opposite the move and scaled by capped relative volume.
relvol=v.div(v.rolling(20,min_periods=10).median()).clip(upper=5)
f=(-p.pct_change(5)*relvol).replace([np.inf,-np.inf],np.nan)
print('FACTOR volume_confirmed_short_reversal_5d VALIDATED_THROUGH',CUT.date())
print('definition=negative 5d return times own volume / prior 20d median volume, relative-volume capped at 5')
print('assets=%d factor_dates=%d cells=%d coverage=%.6f volume_coverage=%.6f'%(len(A),f.notna().any(axis=1).sum(),f.notna().sum().sum(),f.notna().mean().mean(),v.notna().mean().mean()))
ics={}
for h in [1,5,10,20]:
 y=p.shift(-h).div(p)-1; out=[]; ns=[]
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   x=spearmanr(q.f,q.y).statistic
   if np.isfinite(x): out.append((d,x));ns.append(len(q))
 s=pd.Series(dict(out),dtype=float);ics[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),s.mean()/s.std(ddof=1),len(s),(s>0).mean(),np.mean(ns)))
for n,lo,hi in [('2020_24','2020-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_34','2027-01-01',CUT)]:
 s=ics[1].loc[lo:hi]
 print('REGIME1 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(n,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turns),len(turns)))
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
 z=pd.read_pickle(max(hits,key=os.path.getmtime));q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
 if len(q)<8 or q.a.nunique()<2 or q.b.nunique()<2:missing.append(fid);continue
 scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE effective=%d evidence=%d missing=%d'%(len(eff),len(scores),len(missing)))
if len(scores)==len(eff):print('MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else:print('MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
f.to_pickle('scripts/miner_3_20340119_volume_confirmed_short_reversal_5d_signal.pkl')
