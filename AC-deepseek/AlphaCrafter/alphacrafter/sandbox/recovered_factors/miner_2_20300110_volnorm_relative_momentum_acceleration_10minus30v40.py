"""Miner 2: validate volatility-normalized relative momentum acceleration (one idea)."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-01-09')
px=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date')['close'] for a in ASSETS}).sort_index().loc[:END]
px=px.where(px>0); r=px.pct_change()
# Acceleration: short 10d return minus the preceding 30d return, normalized by 40d daily volatility.
# Cross-sectional demeaning removes a common global risk-on/off impulse.
m10=px.pct_change(10); prior30=px.shift(10)/px.shift(40)-1
f=(m10-prior30).div(r.rolling(40,min_periods=30).std()).replace([np.inf,-np.inf],np.nan)
f=f.sub(f.median(axis=1),axis=0)
def calc(h):
 y=px.shift(-h).div(px)-1; rows=[]
 for dt in f.index:
  ok=f.loc[dt].notna()&y.loc[dt].notna()
  if ok.sum()>=8: rows.append((dt,spearmanr(f.loc[dt,ok],y.loc[dt,ok]).statistic,ok.sum()))
 z=pd.DataFrame(rows,columns=['date','ic','n'])
 def sm(q):
  sd=q.ic.std(ddof=1)
  return {'dates':len(q),'ic':q.ic.mean(),'icir':q.ic.mean()/sd if len(q)>1 and sd else np.nan,'hit_ratio':(q.ic>0).mean(),'mean_valid_instruments':q.n.mean()}
 return z,sm(z)
print('FACTOR volnorm_relative_momentum_acceleration_10minus30v40; visible_cutoff',END.date(),'panel_dates',len(f),'assets',len(ASSETS))
print('COVERAGE',float(f.notna().mean().mean()),'MEAN_NAMES',float(f.notna().sum(axis=1).mean()))
M={}
for h in (1,5,10,20):
 z,s=calc(h); M[h]=s; print('H',h,json.dumps(s))
 for label,lo,hi in [('2020_22','2020-01-01','2022-12-31'),('2023_24','2023-01-01','2024-12-31'),('2025_26','2025-01-01','2026-12-31'),('2027_28','2027-01-01','2028-12-31'),('2029','2029-01-01','2030-01-09')]:
  q=z[(z.date>=lo)&(z.date<=hi)]; sd=q.ic.std(ddof=1); print('REGIME',h,label,'dates',len(q),'ic',q.ic.mean(),'icir',q.ic.mean()/sd if len(q)>1 and sd else np.nan,'hit',(q.ic>0).mean())
valid=f.dropna(thresh=8); st=[]
for i in range(1,len(valid)):
 ok=valid.iloc[i].notna()&valid.iloc[i-1].notna()
 if ok.sum()>=8: st.append(spearmanr(valid.iloc[i,ok],valid.iloc[i-1,ok]).statistic)
print('TURNOVER rank_stability',np.mean(st),'implied_daily', (1-np.mean(st))/2)
# Full library audit: only exact/recoverable factor signal panels count as evidence.
eff=[]
for p in glob.glob('factors/*.json'):
 try:
  d=json.load(open(p));
  if d.get('validation',{}).get('status')=='EFFECTIVE': eff.append((p,d.get('factor_id','')))
 except: pass
mx=-1; who=None; found=[]; missing=[]
for p,fid in eff:
 candidates=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not candidates: missing.append(fid); continue
 try:
  g=pd.read_pickle(candidates[0]).reindex(index=f.index,columns=ASSETS)
  vals=[]
  for dt in f.index:
   ok=f.loc[dt].notna()&g.loc[dt].notna()
   if ok.sum()>=8: vals.append(spearmanr(f.loc[dt,ok],g.loc[dt,ok]).statistic)
  if not vals: missing.append(fid); continue
  rho=float(np.nanmean(vals)); found.append(fid)
  if abs(rho)>mx: mx=abs(rho);who=fid
 except Exception: missing.append(fid)
print('AUDIT',json.dumps({'effective_factor_count':len(eff),'artifacts_resolved':len(found),'missing':missing,'complete':not missing,'max_abs_library_correlation':mx if not missing else None,'observed_max_abs_library_correlation':mx if mx>=0 else None,'most_correlated':who}))
f.to_pickle('scripts/miner_2_20300110_volnorm_relative_momentum_acceleration_10minus30v40_signal.pkl')
