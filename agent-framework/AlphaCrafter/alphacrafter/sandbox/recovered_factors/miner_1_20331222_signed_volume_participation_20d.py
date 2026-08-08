"""One candidate: 20-session signed-volume participation, validated only through completed 2033-12-21."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
CUT=pd.Timestamp('2033-12-21')
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
frames={}
for a in A:
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
 frames[a]=d
p=pd.DataFrame({a:d['close'] for a,d in frames.items()}); r=p.pct_change()
v=pd.DataFrame({a:pd.to_numeric(d['volume'],errors='coerce') if 'volume' in d else np.nan for a,d in frames.items()})
# Mean return weighted by each day's relative volume: positive if high-volume days tend to be up days.
lv=np.log(v.replace(0,np.nan)); relv=lv-lv.rolling(20,min_periods=16).mean()
f=(r*relv).rolling(20,min_periods=16).mean().replace([np.inf,-np.inf],np.nan)
print('FACTOR signed_volume_participation_20d VALIDATED_THROUGH',CUT.date())
print('expression=mean_20(return_t*(log(volume_t)-mean_20(log(volume)))); higher means upside moves coincide with above-normal volume')
print('coverage=%.6f calendar_dates=%d valid_cells=%d assets=%d'%(f.notna().mean().mean(),f.notna().any(axis=1).sum(),f.notna().sum().sum(),len(A)))
allstats={}
for h in [1,5,10,20]:
 vals=[]; ns=[]; fw=p.shift(-h).div(p)-1
 for d in f.index:
  q=pd.concat([f.loc[d].rename('f'),fw.loc[d].rename('y')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1 and q.y.nunique()>1:
   z=spearmanr(q.f,q.y).statistic
   if np.isfinite(z): vals.append((d,z));ns.append(len(q))
 s=pd.Series(dict(vals),dtype=float); ir=s.mean()/s.std(ddof=1)
 allstats[h]=s
 print('H%d IC=%+.6f ICIR=%+.6f dates=%d hit=%.4f meanN=%.2f'%(h,s.mean(),ir,len(s),(s>0).mean(),np.mean(ns)))
for name,lo,hi in [('2020_2024','2020-01-01','2024-12-31'),('2025_2026','2025-01-01','2026-12-31'),('2027_2033','2027-01-01',CUT)]:
 s=allstats[20].loc[lo:hi]; print('REGIME20 %s dates=%d IC=%+.6f ICIR=%+.6f hit=%.4f'%(name,len(s),s.mean(),s.mean()/s.std(ddof=1),(s>0).mean()))
rk=f.rank(axis=1,pct=True); turns=[]
for i in range(1,len(rk)):
 q=rk.iloc[[i-1,i]].T.dropna()
 if len(q)>=8: turns.append(1-spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('turnover=%.6f pairs=%d'%(np.mean(turns),len(turns)))
eff=[]
for fn in glob.glob('factors/*.json'):
 try:
  x=json.load(open(fn));
  if x.get('validation',{}).get('status')=='EFFECTIVE': eff.append(x['factor_id'])
 except Exception: pass
found=[]; scores=[]
for fid in eff:
 hits=glob.glob('scripts/*_'+fid+'_signal.pkl')
 if hits:
  found.append(fid); z=pd.read_pickle(max(hits,key=os.path.getmtime)); q=pd.concat([f.stack().rename('a'),z.stack().rename('b')],axis=1).dropna()
  if len(q)>=8 and q.a.nunique()>1 and q.b.nunique()>1: scores.append((abs(spearmanr(q.a,q.b).statistic),fid,len(q)))
print('INDEPENDENCE artifacts=%d effective=%d missing=%d'%(len(found),len(eff),len(eff)-len(found)))
if scores: print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION=%.6f factor=%s cells=%d'%max(scores))
else: print('PARTIAL_MAX_ABS_LIBRARY_CORRELATION=UNAVAILABLE')
print('ADMISSION requires complete artifact evidence and max correlation <0.5000.')
f.to_pickle('scripts/miner_1_20331222_signed_volume_participation_20d_signal.pkl')
