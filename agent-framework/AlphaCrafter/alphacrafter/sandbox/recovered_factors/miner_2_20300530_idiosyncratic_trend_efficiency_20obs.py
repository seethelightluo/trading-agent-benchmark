"""miner_2 20300530: cross-asset idiosyncratic trend efficiency.
Tests whether a 20-session trend that is weakly coupled to the contemporaneous
rest-of-universe benchmark is more likely to persist. The signal is 20d return
multiplied by one minus absolute rolling correlation to the equal-weight return
of the other available tradable instruments. It separates idiosyncratic trends
from common risk-on/risk-off moves using price data only."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-05-29')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index()
r=np.log(C).diff()
# Excluding the asset itself avoids mechanical self-correlation.
others=pd.DataFrame({a:r.drop(columns=a).mean(axis=1) for a in A})
corr=pd.DataFrame({a:r[a].rolling(20,min_periods=15).corr(others[a]) for a in A})
trend=np.log(C/C.shift(20))
F=(trend*(1-corr.abs())).loc[:END]
def metrics(h):
 fut=(C.shift(-h)/C-1).reindex(F.index); rec=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fut.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: rec.append((dt,float(spearmanr(z.f,z.y).statistic)));ns.append(len(z))
 ic=pd.Series(dict(rec),dtype=float); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd) if sd else None,'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(ns))}
ALL={}
for h in (1,5,10,20):
 ic,ALL[h]=metrics(h);print('HORIZON',h,json.dumps(ALL[h],sort_keys=True))
ic,_=metrics(5)
for label,mask in [('2020_2021',ic.index.year<=2021),('2022_2023',ic.index.year.isin([2022,2023])),('2024_2026',ic.index.year.isin([2024,2025,2026])),('2027_2030',ic.index.year>=2027)]:
 x=ic[mask];print('REGIME_5D',label,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()) if len(x) else None)
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
 except Exception:pass
files=glob.glob('scripts/*_signal.pkl');ev={};mx=0.;mxf=None
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 ms=[p for p in files if key in os.path.basename(p)]
 if not ms: ev[fid]={'rho':None,'common_signal_cells':0,'file':None};mx=np.inf;print('LIBRARY_CORR',fid,'MISSING');continue
 p=max(ms,key=os.path.getmtime)
 try:
  lib=pd.read_pickle(p).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('candidate'),lib.stack().rename('library')],axis=1).dropna()
  rho=float(spearmanr(z.candidate,z.library).statistic) if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();rho=np.nan
 ev[fid]={'rho':rho if np.isfinite(rho) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(rho):mx=np.inf
 elif abs(rho)>mx:mx=abs(rho);mxf=fid
 print('LIBRARY_CORR',fid,'cells',len(z),'spearman',rho)
print('FACTOR idiosyncratic_trend_efficiency_20obs')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in ALL.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',mxf,'AUDITED',len(active),'EVIDENCE',json.dumps(ev,sort_keys=True))
F.to_pickle('scripts/miner_2_20300530_idiosyncratic_trend_efficiency_20obs_signal.pkl')
