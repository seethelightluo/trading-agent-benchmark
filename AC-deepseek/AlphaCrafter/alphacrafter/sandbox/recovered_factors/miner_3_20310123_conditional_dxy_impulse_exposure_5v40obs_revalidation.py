"""miner_3 revalidation: conditional DXY impulse exposure, data through 2031-01-22."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-01-22')
def close(path): return pd.read_csv(path,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index()
dxy=close('../persistent/index_data/DXY.csv').reindex(C.index).ffill()
r=np.log(C).diff(); dr=np.log(dxy).diff()
beta=r.rolling(40,min_periods=30).cov(dr).div(dr.rolling(40,min_periods=30).var(),axis=0)
F=beta.mul(-dr.rolling(5,min_periods=5).sum(),axis=0).loc[:END]
def calc(h):
 fut=(C.shift(-h)/C-1).reindex(F.index); rec=[]; widths=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fut.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   v=spearmanr(z.f,z.y).statistic
   if np.isfinite(v): rec.append((dt,float(v))); widths.append(len(z))
 ic=pd.Series(dict(rec),dtype=float); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(widths))}
ALL={}
for h in (1,5,10,20):
 ic,ALL[h]=calc(h); print('HORIZON',h,json.dumps(ALL[h],sort_keys=True))
for h in (5,20):
 ic,_=calc(h)
 for label,mask in [('2020_2021',ic.index.year<=2021),('2022_2023',ic.index.year.isin([2022,2023])),('2024_2026',ic.index.year.isin([2024,2025,2026])),('2027_2030',ic.index.year.isin([2027,2028,2029,2030])),('2031_ytd',ic.index.year==2031)]:
  x=ic[mask]; print('REGIME',h,label,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()) if len(x) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q): st.append(float(q))
active=[]
for p in glob.glob('factors/*.json'):
 d=json.load(open(p))
 if d.get('validation',{}).get('status')=='EFFECTIVE': active.append(d['factor_id'])
evidence={}; mx=0.; most=None; complete=True
for fid in active:
 # exclude own stale signal: correlation with itself is irrelevant for revalidation
 if fid=='miner_3_conditional_dxy_impulse_exposure_5v40obs': continue
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 matches=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not matches: evidence[fid]={'rho':None,'common_signal_cells':0};complete=False;print('LIBRARY_CORR',fid,'MISSING');continue
 p=max(matches,key=os.path.getmtime)
 try:
  lib=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('candidate'),lib.stack().rename('library')],axis=1).dropna();q=spearmanr(z.candidate,z.library).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 rho=float(q) if np.isfinite(q) else None;evidence[fid]={'rho':rho,'common_signal_cells':len(z),'file':p}
 if rho is None: complete=False
 else:
  if abs(rho)>mx:mx=abs(rho);most=fid
 print('LIBRARY_CORR',fid,'cells',len(z),'spearman',rho)
print('FACTOR conditional_dxy_impulse_exposure_5v40obs REVALIDATION')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in ALL.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(evidence,sort_keys=True))
F.to_pickle('scripts/miner_3_20310123_conditional_dxy_impulse_exposure_5v40obs_revalidation_signal.pkl')
