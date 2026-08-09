"""miner_3 20300502: downside intraday range pressure.
Tests whether assets whose recent trading ranges are disproportionately realized on down-close
sessions subsequently mean-revert cross-sectionally. Uses only asset OHLC available at each date."""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-05-01')
def load(a,field):
 p='../persistent/stock_data/'+a+'.csv'
 d=pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d[field].astype(float)
C=pd.DataFrame({a:load(a,'close') for a in A}).sort_index()
H=pd.DataFrame({a:load(a,'high') for a in A}).sort_index(); L=pd.DataFrame({a:load(a,'low') for a in A}).sort_index()
r=np.log(C).diff(); rg=(H-L).div(C.replace(0,np.nan))
# Higher means an unusually large fraction of the trailing ten-session range occurred on down days.
F=(rg.where(r<0,0).rolling(10,min_periods=8).sum()/rg.rolling(10,min_periods=8).sum()).loc[:END]
def metrics(h):
 fut=(C.shift(-h)/C-1).reindex(F.index); out=[]; n=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fut.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8: out.append((dt,float(spearmanr(z.f,z.y).statistic)));n.append(len(z))
 ic=pd.Series(dict(out),dtype=float); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd) if sd else None,'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(n))}
ALL={}
for h in (1,5,10,20):
 ic,ALL[h]=metrics(h);print('HORIZON',h,json.dumps(ALL[h],sort_keys=True))
ic,_=metrics(5)
for label, mask in [('2020_2021',ic.index.year<=2021),('2022_2023',ic.index.year.isin([2022,2023])),('2024_2026',ic.index.year.isin([2024,2025,2026])),('2027_2030',ic.index.year>=2027)]:
 x=ic[mask]; print('REGIME_5D',label,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()) if len(x) else None)
# rank persistence / turnover
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
# Compare against ALL admitted factors. Existing pkl convention is inspected by factor-id token matching.
active=[]
for fp in glob.glob('factors/*.json'):
 if fp.endswith('.bak') or '_deprecated' in fp: continue
 d=json.load(open(fp)); active.append(d['factor_id'])
files=glob.glob('scripts/*_signal.pkl'); evidence={}; mx=0.0
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 matches=[p for p in files if key in os.path.basename(p)]
 if not matches:
  evidence[fid]={'rho':None,'common_signal_cells':0,'file':None};mx=np.inf;print('LIBRARY_CORR',fid,'MISSING');continue
 p=max(matches,key=os.path.getmtime)
 try: lib=pd.read_pickle(p).reindex(index=F.index,columns=A)
 except Exception as e: evidence[fid]={'rho':None,'common_signal_cells':0,'file':p};mx=np.inf;print('LIBRARY_CORR',fid,'UNREADABLE');continue
 z=pd.concat([F.stack().rename('candidate'),lib.stack().rename('library')],axis=1).dropna()
 rho=float(spearmanr(z.candidate,z.library).statistic) if len(z)>=8 else np.nan
 evidence[fid]={'rho':rho,'common_signal_cells':len(z),'file':p};mx=max(mx,abs(rho)) if np.isfinite(rho) else np.inf
 print('LIBRARY_CORR',fid,'cells',len(z),'spearman',rho)
print('FACTOR downside_intraday_range_pressure_10d')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in ALL.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'EVIDENCE',json.dumps(evidence,sort_keys=True))
F.to_pickle('scripts/miner_3_20300502_downside_intraday_range_pressure_10d_signal.pkl')
