"""miner_2 20300516: drawdown-recovery persistence, one price-only factor idea.
High scores identify assets recovering sharply over five sessions while still below their trailing
60-session high, scaled by trailing realized volatility. Tests whether incomplete recoveries persist."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2030-05-15')
def series(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d['close'].astype(float)
C=pd.DataFrame({a:series(a) for a in A}).sort_index(); r=np.log(C).diff()
vol=r.rolling(20,min_periods=15).std(); dd=C/C.rolling(60,min_periods=45).max()-1
# Strong five-day rebound that remains an incomplete recovery: r5 / vol20 times absolute drawdown.
F=((C/C.shift(5)-1)/vol*(-dd)).replace([np.inf,-np.inf],np.nan).loc[:END]
def get(h):
 y=(C.shift(-h)/C-1).reindex(F.index); vals=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: vals.append((dt,float(spearmanr(z.f,z.y).statistic)));ns.append(len(z))
 ic=pd.Series(dict(vals)); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(ns))}
ALL={}
for h in [1,5,10,20]:
 ic,ALL[h]=get(h);print('HORIZON',h,json.dumps(ALL[h],sort_keys=True))
ic,_=get(5)
for n,m in [('2020_2021',ic.index.year<=2021),('2022_2023',ic.index.year.isin([2022,2023])),('2024_2026',ic.index.year.isin([2024,2025,2026])),('2027_2030',ic.index.year>=2027)]:
 x=ic[m]; print('REGIME_5D',n,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()) if len(x) else None)
st=[]
for t in range(1,len(F)):
 z=pd.concat([F.iloc[t-1],F.iloc[t]],axis=1).dropna()
 if len(z)>=8: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
# Explicit signal artifact mapping: require one usable artifact for every active factor.
active=[]
for p in glob.glob('factors/*.json'):
 if p.endswith('.bak') or '_deprecated' in p: continue
 d=json.load(open(p));
 if d.get('validation',{}).get('status')=='EFFECTIVE': active.append(d['factor_id'])
files=glob.glob('scripts/*_signal.pkl'); ev={}; mx=0.; missing=0
for fid in active:
 # factor-id convention is miner_N_<descriptive id>; find artifact containing descriptive id.
 key=fid.split('_',2)[-1]; hits=[p for p in files if key in os.path.basename(p)]
 if not hits:
  ev[fid]={'rho':None,'common_signal_cells':0,'file':None};missing+=1;continue
 p=max(hits,key=os.path.getmtime)
 try: q=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('candidate'),q.stack().rename('library')],axis=1).dropna(); rho=float(spearmanr(z.candidate,z.library).statistic) if len(z)>=8 else np.nan
 except Exception: rho=np.nan;z=pd.DataFrame()
 ev[fid]={'rho':rho if np.isfinite(rho) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(rho): missing+=1
 else: mx=max(mx,abs(rho))
 print('LIBRARY_CORR',fid,'cells',len(z),'spearman',rho)
print('FACTOR volatility_scaled_incomplete_drawdown_recovery_persistence_5x20x60')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in ALL.items()},sort_keys=True));print('LIBRARY_ACTIVE',len(active),'MISSING_EVIDENCE',missing,'MAX_ABS_LIBRARY_CORRELATION',mx if missing==0 else None,'EVIDENCE',json.dumps(ev,sort_keys=True))
F.to_pickle('scripts/miner_2_20300516_volatility_scaled_incomplete_drawdown_recovery_persistence_5x20x60_signal.pkl')
