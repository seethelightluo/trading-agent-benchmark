"""miner_2: standardized cross-asset median-residual direction persistence (20 observations).
For each asset, subtract the daily contemporaneous median cross-asset return, then
compute the trailing 20-session mean residual divided by its trailing residual standard
deviation. This distinguishes persistent relative direction from residual amplitude.
All factor values at d use closes through d; forward returns start after d.
"""
import os, glob, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-02-19')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=C.pct_change(); E=R.sub(R.median(axis=1),axis=0)
# Relative directional persistence, standardized solely by trailing own residual variability.
F=(E.rolling(20,min_periods=20).mean()/E.rolling(20,min_periods=20).std(ddof=1)).loc[:END].replace([np.inf,-np.inf],np.nan)
def measure(h):
 y=(C.shift(-h)/C-1).reindex(F.index); rows=[]; widths=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('r')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.r).statistic
   if np.isfinite(q): rows.append((d,float(q))); widths.append(len(z))
 ic=pd.Series(dict(rows)); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(widths))}
metrics={}
for h in (1,5,10,20):
 ic,m=measure(h); metrics[str(h)]=m; print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for lab,yrs in [('2020_2021',[2020,2021]),('2022_2023',[2022,2023]),('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031_ytd',[2031])]:
   x=ic[ic.index.year.isin(yrs)]; print('REGIME_10D',lab,json.dumps({'dates':len(x),'ic':float(x.mean()) if len(x) else None,'icir':float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit':float((x>0).mean()) if len(x) else None}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
def key(fid):
 for prefix in ('miner_1_','miner_2_','miner_3_'):
  if fid.startswith(prefix): return fid[len(prefix):]
 return fid
active=[]
for p in glob.glob('factors/*.json'):
 try:
  d=json.load(open(p))
  if d.get('validation',{}).get('status')=='EFFECTIVE': active.append(d['factor_id'])
 except Exception: pass
mx=0.;who=None;evidence={};complete=True
for fid in active:
 matches=glob.glob('scripts/*'+key(fid)+'*signal.pkl')
 if not matches:
  complete=False;evidence[fid]={'rho':None,'common_signal_cells':0};print('LIBRARY_CORR',fid,'MISSING');continue
 p=max(matches,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('candidate'),L.stack().rename('library')],axis=1).dropna(); q=float(spearmanr(z.candidate,z.library).statistic) if len(z)>=8 else np.nan
 except Exception: q=np.nan;z=pd.DataFrame()
 evidence[fid]={'rho':q if np.isfinite(q) else None,'common_signal_cells':len(z),'file':os.path.basename(p)}
 print('LIBRARY_CORR',fid,'cells',len(z),'rho',q)
 if not np.isfinite(q): complete=False
 elif abs(q)>mx: mx=abs(q);who=fid
print('SUMMARY',json.dumps({'factor':'standardized_cross_asset_median_residual_direction_persistence_20obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'implied_rank_turnover':float(1-np.mean(st)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'metrics':metrics,'evidence':evidence},sort_keys=True))
F.to_pickle('scripts/miner_2_20310220_standardized_cross_asset_median_residual_direction_persistence_20obs_signal.pkl')
