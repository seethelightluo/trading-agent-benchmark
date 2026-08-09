"""One candidate: inverse 20-session directional participation (breadth exhaustion).
Signal is the negative of directional daily-return breadth.  It tests whether assets
with unusually one-sided 20-session participation subsequently mean revert."""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-10-27')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}); R=np.log(C).diff()
pos=(R>0).rolling(20,min_periods=15).sum(); neg=(R<0).rolling(20,min_periods=15).sum()
F=(-((pos-neg)/(pos+neg))).loc[:END]
def one_ic(h):
 y=(C.shift(-h)/C-1).reindex(F.index); out=[]; breadth=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('factor'),y.loc[d].rename('forward')],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.factor,z.forward).statistic
   if np.isfinite(v): out.append((d,v)); breadth.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':int(len(s)),'mean_valid_instruments':float(np.mean(breadth))}
for h in (1,5,10,20): print('HORIZON',h,json.dumps(one_ic(h)[1],sort_keys=True))
s,_=one_ic(1)
for name,mask in [('2020_2023',s.index.year<=2023),('2024_2027',(s.index.year>=2024)&(s.index.year<=2027)),('2028_2030',(s.index.year>=2028)&(s.index.year<=2030)),('2031_2032',s.index.year>=2031),('recent_6m',s.index>=END-pd.Timedelta(days=183))]:
 q=s[mask]; print('REGIME',name,json.dumps({'ic_dates':len(q),'ic':float(q.mean()),'icir':float(q.mean()/q.std(ddof=1)),'hit_ratio':float((q>0).mean())}))
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except Exception: pass
complete=True; mx=-1.; who=None
for fid in active:
 paths=[p for p in glob.glob('scripts/*signal.pkl') if fid in os.path.basename(p)]
 if not paths: print('CORRELATION',fid,'MISSING_ARTIFACT'); complete=False; continue
 L=pd.read_pickle(max(paths,key=os.path.getmtime)).reindex(index=F.index,columns=A)
 z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna()
 rho=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 print('CORRELATION',fid,'n=',len(z),'rho=',rho)
 if not np.isfinite(rho): complete=False
 elif abs(rho)>mx: mx,who=abs(rho),fid
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('SUMMARY',json.dumps({'validation_cutoff':str(END.date()),'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'rank_stability_1d':float(np.nanmean(st)),'implied_rank_turnover':float(1-np.nanmean(st)),'effective_library_count':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':float(mx) if complete else None,'most_correlated':who},sort_keys=True))
F.to_pickle('scripts/miner_2_20321028_inverse_directional_participation_exhaustion_20obs_signal.pkl')
