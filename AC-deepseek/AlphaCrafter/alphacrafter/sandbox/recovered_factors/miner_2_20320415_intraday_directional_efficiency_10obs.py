"""miner_2: intraday directional efficiency (10 observations), cutoff 2032-04-14.
The signal is signed net close-to-open movement divided by gross intraday movement.
It separates persistent intraday pressure from choppy, offsetting intraday action.
"""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-04-14')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d[['open','close']].astype(float)
D={a:load(a) for a in A}
O=pd.DataFrame({a:D[a]['open'] for a in A}); C=pd.DataFrame({a:D[a]['close'] for a in A}).reindex(O.index)
r=np.log(C/O)
# Require a meaningful amount of intraday movement; denominator prevents scale bias.
F=r.rolling(10,min_periods=8).sum()/r.abs().rolling(10,min_periods=8).sum().clip(lower=1e-8)
F=F.loc[:END]
def metrics(X,h):
 y=(C.shift(-h)/C-1).reindex(X.index); vals=[];ns=[]
 for d in X.index:
  z=pd.concat([X.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): vals.append((d,float(q)));ns.append(len(z))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for name,X in [('continuation',F),('reversal',-F)]:
 for h in (1,5,10,20): print('HORIZON',name,h,json.dumps(metrics(X,h)[1],sort_keys=True))
 s,_=metrics(X,1)
 for lab,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year.isin([2027,2028,2029,2030])),('2031_2032',s.index.year>=2031),('recent_6m',s.index>=END-pd.Timedelta(days=183))]:
  q=s[mask];print('REGIME',name,lab,len(q),None if len(q)==0 else float(q.mean()),None if len(q)<2 else float(q.mean()/q.std(ddof=1)),None if len(q)==0 else float((q>0).mean()))
# stability and complete library audit
rs=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):rs.append(q)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except Exception:pass
complete=True; mx=0;who=None
for fid in active:
 matches=[p for p in glob.glob('scripts/*_signal.pkl') if fid in os.path.basename(p)]
 if not matches:
  complete=False;print('LIBRARY_CORR',fid,'MISSING');continue
 try:
  L=pd.read_pickle(max(matches,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna();q=spearmanr(z.x,z.l).statistic if len(z)>=8 else np.nan
 except Exception:q=np.nan;z=pd.DataFrame()
 print('LIBRARY_CORR',fid,len(z),q)
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=float(abs(q));who=fid
print('SUMMARY',json.dumps({'factor_id':'miner_2_intraday_directional_efficiency_10obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_valid_instruments':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(rs)),'implied_rank_turnover':float(1-np.mean(rs)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'most_correlated':who},sort_keys=True))
F.to_pickle('scripts/miner_2_20320415_intraday_directional_efficiency_10obs_signal.pkl')
