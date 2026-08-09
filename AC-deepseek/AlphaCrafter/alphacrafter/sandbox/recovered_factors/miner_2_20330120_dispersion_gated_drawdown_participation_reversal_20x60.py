"""One candidate: cross-asset-dispersion-gated drawdown participation reversal.
Tests whether drawdown/participation exhaustion is more reliable during unusually
dispersed cross-asset sessions, when relative stress is informative rather than
merely a common market move.
"""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2033-01-19')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}); R=np.log(C).diff()
pos=(R>0).rolling(20,min_periods=16).sum(); neg=(R<0).rolling(20,min_periods=16).sum()
part=(pos-neg)/(pos+neg)
dd=1-C/C.rolling(60,min_periods=45).max(); vol=R.rolling(20,min_periods=16).std().replace(0,np.nan)
stress=(dd/vol).clip(upper=15)
# Cross-sectional dispersion of daily returns; activate only above own trailing median.
disp=R.std(axis=1,skipna=True); gate=disp>=disp.rolling(60,min_periods=45).median()
# Individual stress percentile prevents a common drawdown from dominating all assets.
stress_rank=stress.rank(axis=1,pct=True)
F=(-part*(1+stress_rank)).where(gate, np.nan).loc[:END]
def one_ic(h):
 y=(C.shift(-h)/C-1).reindex(F.index); out=[];breadth=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('factor'),y.loc[d].rename('forward')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.factor,z.forward).statistic
   if np.isfinite(q):out.append((d,q));breadth.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(breadth))}
for h in (1,5,10,20): print('HORIZON',h,json.dumps(one_ic(h)[1],sort_keys=True))
s,_=one_ic(1)
for name,mask in [('2020_2023',s.index.year<=2023),('2024_2027',(s.index.year>=2024)&(s.index.year<=2027)),('2028_2030',(s.index.year>=2028)&(s.index.year<=2030)),('2031_2032',(s.index.year>=2031)&(s.index.year<=2032)),('recent_6m',s.index>=END-pd.Timedelta(days=183))]:
 q=s[mask]; print('REGIME',name,json.dumps({'ic_dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,'hit_ratio':float((q>0).mean()) if len(q) else None}))
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except Exception: pass
complete=True;mx=-1;who=None;found=0
for fid in active:
 paths=[p for p in glob.glob('scripts/*signal.pkl') if fid in os.path.basename(p)]
 if not paths: complete=False; continue
 L=pd.read_pickle(max(paths,key=os.path.getmtime)).reindex(index=F.index,columns=A)
 z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna()
 rho=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 if not np.isfinite(rho): complete=False
 else:
  found+=1
  if abs(rho)>mx:mx,who=abs(rho),fid
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('SUMMARY',json.dumps({'validation_cutoff':str(END.date()),'panel_dates':len(F),'active_signal_dates':int(gate.loc[F.index].sum()),'coverage':float(F.notna().mean().mean()),'rank_stability_1d':float(np.nanmean(st)),'implied_rank_turnover':float(1-np.nanmean(st)),'effective_library_count':len(active),'library_artifacts_found':found,'correlation_evidence_complete':complete,'max_abs_library_correlation':float(mx) if complete else None,'most_correlated':who},sort_keys=True))
F.to_pickle('scripts/miner_2_20330120_dispersion_gated_drawdown_participation_reversal_20x60_signal.pkl')
