"""miner_2 20300627: dispersion-conditioned relative strength.
A 20-session asset return relative to the cross-asset median is multiplied by
whether cross-sectional 20d-return dispersion is high or low versus its own
trailing 60-session history. This tests momentum in differentiated markets and
contrarian ranking when common-factor markets suppress dispersion."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-06-26')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R20=np.log(C/C.shift(20))
# cross-sectional dispersion and a strictly historical standardized regime score
D=R20.std(axis=1,ddof=1); dm=D.rolling(60,min_periods=40).mean(); ds=D.rolling(60,min_periods=40).std()
reg=np.tanh((D-dm)/ds).replace([np.inf,-np.inf],np.nan)
# Relative strength only; scalar regime score switches its ordering in low dispersion.
F=R20.sub(R20.median(axis=1),axis=0).mul(reg,axis=0).loc[:END]
def metrics(h):
 fut=(C.shift(-h)/C-1).reindex(F.index); rec=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fut.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):rec.append((dt,float(q)));ns.append(len(z))
 ic=pd.Series(dict(rec),dtype=float); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd) if sd else None,'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(ns))}
ALL={}
for h in (1,5,10,20):
 ic,ALL[h]=metrics(h);print('HORIZON',h,json.dumps(ALL[h],sort_keys=True))
ic,_=metrics(5)
for label,mask in [('2020_2021',ic.index.year<=2021),('2022_2023',ic.index.year.isin([2022,2023])),('2024_2026',ic.index.year.isin([2024,2025,2026])),('2027_2030',ic.index.year>=2027)]:
 x=ic[mask]; print('REGIME_5D',label,'dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
active=[]
for fp in glob.glob('factors/*.json'):
 if fp.endswith('.bak') or '_deprecated' in fp:continue
 try:
  d=json.load(open(fp))
  if d.get('validation',{}).get('status')=='EFFECTIVE':active.append(d['factor_id'])
 except Exception: pass
files=glob.glob('scripts/*_signal.pkl'); ev={};mx=0.;mxf=None
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_',''); ms=[p for p in files if key in os.path.basename(p)]
 if not ms: ev[fid]={'rho':None,'common_signal_cells':0,'file':None};mx=np.inf;print('LIBRARY_CORR',fid,'MISSING');continue
 p=max(ms,key=os.path.getmtime)
 try:
  lib=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('candidate'),lib.stack().rename('library')],axis=1).dropna();rho=float(spearmanr(z.candidate,z.library).statistic) if len(z)>=8 else np.nan
 except Exception:z=pd.DataFrame();rho=np.nan
 ev[fid]={'rho':rho if np.isfinite(rho) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(rho):mx=np.inf
 elif abs(rho)>mx:mx=abs(rho);mxf=fid
 print('LIBRARY_CORR',fid,'cells',len(z),'spearman',rho)
print('FACTOR dispersion_conditioned_relative_strength_20x60obs')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in ALL.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',mxf,'AUDITED',len(active),'EVIDENCE',json.dumps(ev,sort_keys=True))
F.to_pickle('scripts/miner_2_20300627_dispersion_conditioned_relative_strength_20x60obs_signal.pkl')
