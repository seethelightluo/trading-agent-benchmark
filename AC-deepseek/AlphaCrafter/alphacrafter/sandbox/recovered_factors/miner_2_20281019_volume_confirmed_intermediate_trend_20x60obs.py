"""One candidate: volume-confirmed intermediate trend persistence (20x60 observations).
A positive score requires a 20-observation return supported by above-baseline
trading participation; volumes with zero/missing observations are unavailable rather
than imputed.  This tests whether participation distinguishes durable cross-asset
trend from unconfirmed price movement.  Uses only data through 2028-10-18.
"""
import glob,json,os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ASSETS=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-10-18')
def field(a,c):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,c].astype(float)
P=pd.DataFrame({a:field(a,'close') for a in ASSETS}).sort_index()
V=pd.DataFrame({a:field(a,'volume') for a in ASSETS}).reindex(P.index)
r=np.log(P/P.shift())
# Relative participation is log 20d median volume / prior 60d median volume.
# no cross-asset raw-volume comparison, since units differ materially.
validv=V.where(V>0)
part=np.log(validv.rolling(20,min_periods=15).median()/validv.shift(20).rolling(60,min_periods=40).median())
trend=P/P.shift(20)-1
vol=r.rolling(20,min_periods=15).std()*np.sqrt(20)
F=(trend/vol)*part

def metrics(h):
 R=P.shift(-h)/P-1; out=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),R.loc[d].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:
   out.append((d,float(spearmanr(z.f,z.r).statistic))); ns.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
allm={}
for h in (1,5,10,20):
 s,m=metrics(h);allm[h]=m; print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==10:
  for label,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2025',s.index.year.isin([2024,2025])),('2026_2028',s.index.year>=2026)]:
   x=s[mask]; print('REGIME_10D',label,json.dumps({'dates':len(x),'ic':float(x.mean()),'icir':float(x.mean()/x.std(ddof=1)),'hit':float((x>0).mean())}))
stab=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: stab.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('PANEL',json.dumps({'signal_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(stab)),'implied_turnover':float(1-np.mean(stab))}))
# Full library audit. Require evidence for every active factor, and reject if an artifact is absent.
F.to_pickle('scripts/miner_2_20281019_volume_confirmed_intermediate_trend_20x60obs_signal.pkl')
rows=[]; missing=[]
for path in glob.glob('factors/*.json'):
 if path.endswith('.bak'): continue
 try:
  meta=json.load(open(path))
  if meta.get('validation',{}).get('status')!='EFFECTIVE': continue
  fid=meta['factor_id']; hits=glob.glob('scripts/*'+fid+'*signal.pkl')
  if not hits: missing.append(fid); continue
  lib=pd.read_pickle(sorted(hits)[-1]); a,b=F.align(lib,join='inner',axis=0)
  z=pd.DataFrame({'a':a.stack(),'b':b.stack()}).dropna()
  rho=float(spearmanr(z.a,z.b).statistic) if len(z)>2 else float('nan')
  rows.append((fid,len(z),rho))
 except Exception as e: missing.append(os.path.basename(path)+':'+str(e))
for x in rows: print('LIBRARY_CORR',x[0],x[1],x[2])
if missing: print('LIBRARY_MISSING',json.dumps(missing))
else:
 best=max(rows,key=lambda x:abs(x[2])); print('LIBRARY_MAX',json.dumps({'factor':best[0],'cells':best[1],'max_abs_library_correlation':abs(best[2]),'rho':best[2],'audited_factors':len(rows)}))
