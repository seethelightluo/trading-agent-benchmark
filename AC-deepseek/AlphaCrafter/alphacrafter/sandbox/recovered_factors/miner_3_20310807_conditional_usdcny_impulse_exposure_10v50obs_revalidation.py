"""Miner_3 scheduled revalidation of Conditional USDCNY Impulse Exposure, visible through 2031-08-06."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-08-06')
def close(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index()
macro=close('../persistent/index_data/USDCNY.csv').reindex(C.index).ffill(); r=np.log(C).diff(); mr=np.log(macro).diff()
beta=r.rolling(50,min_periods=35).cov(mr).div(mr.rolling(50,min_periods=35).var(),axis=0)
F=beta.mul(-mr.rolling(10,min_periods=10).sum(),axis=0).loc[:END]
def calc(h):
 y=(C.shift(-h)/C-1).reindex(F.index); vals=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): vals.append((d,float(q))); ns.append(len(z))
 ic=pd.Series(dict(vals)); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(ns))}
M={}
for h in (1,5,10,20): _,M[h]=calc(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic20,_=calc(20)
for label,years in [('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031_YTD',[2031])]:
 x=ic20[ic20.index.year.isin(years)];print('REGIME_20D',label,'dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)),'hit',float((x>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna();q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
 if np.isfinite(q):st.append(float(q))
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except:pass
E={};mx=0.;most=None;complete=True
for fid in active:
 if fid=='miner_3_conditional_usdcny_impulse_exposure_10v50obs':continue
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_',''); candidates=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not candidates:E[fid]={'rho':None,'common_signal_cells':0,'file':None};complete=False;continue
 p=max(candidates,key=os.path.getmtime)
 try:
  lib=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('candidate'),lib.stack().rename('library')],axis=1).dropna();q=spearmanr(z.candidate,z.library).statistic if len(z)>=8 else np.nan
 except Exception:z=pd.DataFrame();q=np.nan
 E[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('FACTOR conditional_usdcny_impulse_exposure_10v50obs')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in M.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(E,sort_keys=True))
F.to_pickle('scripts/miner_3_20310807_conditional_usdcny_impulse_exposure_10v50obs_revalidation_signal.pkl')
