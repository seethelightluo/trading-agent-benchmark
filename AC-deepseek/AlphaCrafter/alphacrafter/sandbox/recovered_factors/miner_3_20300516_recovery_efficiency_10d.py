"""miner_3 20300516: recovery efficiency after recent downside.
Higher values indicate that a 10-observation return path has recovered its losses efficiently:
sum(positive log returns)/sum(abs(negative log returns)). This is a path-quality signal,
distinct from endpoint momentum and range-pressure measures."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-05-15')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.close.astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index()
r=np.log(C).diff()
up=r.clip(lower=0).rolling(10,min_periods=8).sum()
down=(-r.clip(upper=0)).rolling(10,min_periods=8).sum()
# Require a non-trivial downside denominator so a perfectly one-sided path is not treated as infinite quality.
F=(up/(down+1e-6)).where(down>0.0005).loc[:END]
def metrics(h):
 fut=(C.shift(-h)/C-1).reindex(F.index); rec=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fut.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: rec.append((dt,float(spearmanr(z.f,z.y).statistic))); ns.append(len(z))
 ic=pd.Series(dict(rec),dtype=float); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(ns))}
ALL={}
for h in (1,5,10,20):
 ic,ALL[h]=metrics(h); print('HORIZON',h,json.dumps(ALL[h],sort_keys=True))
ic,_=metrics(5)
for label,mask in [('2020_2021',ic.index.year<=2021),('2022_2023',ic.index.year.isin([2022,2023])),('2024_2026',ic.index.year.isin([2024,2025,2026])),('2027_2030',ic.index.year>=2027)]:
 x=ic[mask]; print('REGIME_5D',label,'dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)),'hit',float((x>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
active=[]
for fp in glob.glob('factors/*.json'):
 if '_deprecated' not in fp:
  active.append(json.load(open(fp))['factor_id'])
evidence={}; mx=0.0; files=glob.glob('scripts/*_signal.pkl')
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 matches=[p for p in files if key in os.path.basename(p)]
 if not matches:
  evidence[fid]={'rho':None,'common_signal_cells':0,'file':None}; mx=np.inf; print('LIBRARY_CORR',fid,'MISSING'); continue
 p=max(matches,key=os.path.getmtime)
 try: lib=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('candidate'),lib.stack().rename('library')],axis=1).dropna(); rho=float(spearmanr(z.candidate,z.library).statistic) if len(z)>=8 else np.nan
 except Exception: rho=np.nan; z=pd.DataFrame()
 evidence[fid]={'rho':rho if np.isfinite(rho) else None,'common_signal_cells':len(z),'file':p}; mx=max(mx,abs(rho)) if np.isfinite(rho) else np.inf
 print('LIBRARY_CORR',fid,'cells',len(z),'spearman',rho)
print('FACTOR recovery_efficiency_10d')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in ALL.items()},sort_keys=True)); print('MAX_ABS_LIBRARY_CORRELATION',mx,'EVIDENCE',json.dumps(evidence,sort_keys=True))
F.to_pickle('scripts/miner_3_20300516_recovery_efficiency_10d_signal.pkl')
