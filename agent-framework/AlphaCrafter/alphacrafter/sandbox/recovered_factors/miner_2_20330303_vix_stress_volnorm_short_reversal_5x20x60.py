"""One candidate: VIX-stress-conditioned volatility-normalized 5-day reversal.
Tests whether short-horizon cross-asset reversals improve when the observation-only
VIX is high relative to its own trailing 60-session distribution.
"""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2033-03-02')
def close(path):
 return pd.read_csv(path,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A})
R=np.log(C).diff(); vix=close('../persistent/index_data/VIX.csv').reindex(C.index)
# Expanding only through date t: percentile position within the trailing 60 completed VIX closes.
vixpct=vix.rolling(60,min_periods=45).apply(lambda x: pd.Series(x).rank(pct=True).iloc[-1],raw=False)
# Continuous gate: zero below 70th percentile, rising to one at the historical-window maximum.
gate=((vixpct-.70)/.30).clip(0,1)
vol=R.rolling(20,min_periods=16).std().replace(0,np.nan)
F=(-(C.pct_change(5)/vol)).mul(gate,axis=0).loc[:END]
def calc(h):
 y=(C.shift(-h)/C-1).reindex(F.index); vals=[];n=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('r')],axis=1).dropna()
  if len(z)>=8:
   ic=spearmanr(z.f,z.r).statistic
   if np.isfinite(ic): vals.append((d,ic));n.append(len(z))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(n))}
for h in (1,5,10,20):print('HORIZON',h,json.dumps(calc(h)[1],sort_keys=True))
s,_=calc(1)
for name,m in [('2020_2023',s.index.year<=2023),('2024_2027',(s.index.year>=2024)&(s.index.year<=2027)),('2028_2030',(s.index.year>=2028)&(s.index.year<=2030)),('2031_2032',(s.index.year>=2031)&(s.index.year<=2032)),('recent_6m',s.index>=END-pd.Timedelta(days=183))]:
 q=s[m];print('REGIME',name,json.dumps({'ic_dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,'hit_ratio':float((q>0).mean()) if len(q) else None}))
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except: pass
complete=True;found=0;mx=-1;who=None
for fid in active:
 ps=[p for p in glob.glob('scripts/*signal.pkl') if fid in os.path.basename(p)]
 if not ps:complete=False;continue
 try:
  L=pd.read_pickle(max(ps,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna(); rho=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
  if not np.isfinite(rho):complete=False
  else:
   found+=1
   if abs(rho)>mx:mx,who=abs(rho),fid
 except: complete=False
stab=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:stab.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('SUMMARY',json.dumps({'validation_cutoff':str(END.date()),'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'rank_stability_1d':float(np.nanmean(stab)),'implied_rank_turnover':float(1-np.nanmean(stab)),'effective_library_count':len(active),'library_artifacts_found':found,'correlation_evidence_complete':complete,'max_abs_library_correlation':float(mx) if complete else None,'most_correlated':who},sort_keys=True))
F.to_pickle('scripts/miner_2_20330303_vix_stress_volnorm_short_reversal_5x20x60_signal.pkl')
