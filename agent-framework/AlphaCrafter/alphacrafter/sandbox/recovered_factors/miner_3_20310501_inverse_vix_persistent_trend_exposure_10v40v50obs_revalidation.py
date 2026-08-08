"""Scheduled revalidation of inverse VIX persistent trend exposure, data through 2031-04-30."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-04-30')
def close(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index()
vix=close('../persistent/index_data/VIX.csv').reindex(C.index).ffill()
r=np.log(C).diff(); vr=np.log(vix).diff()
beta=r.rolling(50,min_periods=35).cov(vr).div(vr.rolling(50,min_periods=35).var(),axis=0)
imp10=vr.rolling(10,min_periods=10).sum(); trend40=vr.rolling(40,min_periods=40).sum()
F=beta.mul(imp10.where((imp10*trend40)>0),axis=0).loc[:END]
def calc(h):
 future=(C.shift(-h)/C-1).reindex(F.index); out=[]; widths=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),future.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): out.append((d,float(q))); widths.append(len(z))
 ic=pd.Series(dict(out),dtype=float); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(widths))}
allm={}
for h in (1,5,10,20):
 _,allm[h]=calc(h);print('HORIZON',h,json.dumps(allm[h],sort_keys=True))
ic10,_=calc(10)
for lab,mask in [('2024_2026',ic10.index.year.isin([2024,2025,2026])),('2027_2030',ic10.index.year.isin([2027,2028,2029,2030])),('2031_YTD',ic10.index.year==2031)]:
 x=ic10[mask]; print('REGIME_10D',lab,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()) if len(x) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
active=[json.load(open(p))['factor_id'] for p in glob.glob('factors/*.json') if '_deprecated' not in p]
evidence={};mx=0.;most=None
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 m=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not m: evidence[fid]={'rho':None,'common_signal_cells':0};mx=np.inf;print('LIBRARY_CORR',fid,'MISSING');continue
 p=max(m,key=os.path.getmtime)
 try:
  lib=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('candidate'),lib.stack().rename('library')],axis=1).dropna();rho=float(spearmanr(z.candidate,z.library).statistic) if len(z)>=8 else np.nan
 except Exception: rho=np.nan;z=pd.DataFrame()
 evidence[fid]={'rho':rho if np.isfinite(rho) else None,'common_signal_cells':len(z),'file':p}
 if np.isfinite(rho) and abs(rho)>mx: mx=abs(rho);most=fid
 if not np.isfinite(rho): mx=np.inf
 print('LIBRARY_CORR',fid,'cells',len(z),'spearman',rho)
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in allm.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'EVIDENCE_COMPLETE',all(v['rho'] is not None for v in evidence.values()),'N_LIB',len(active))
F.to_pickle('scripts/miner_3_20310501_inverse_vix_persistent_trend_exposure_10v40v50obs_revalidation_signal.pkl')
