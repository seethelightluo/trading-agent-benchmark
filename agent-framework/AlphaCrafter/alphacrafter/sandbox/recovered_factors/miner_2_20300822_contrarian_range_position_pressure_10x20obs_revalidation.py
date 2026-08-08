"""miner_2 20300822 revalidation: contrarian range-position pressure.
Re-tests the admitted 10x20-session price-only range-pressure reversal at the
latest observable cutoff and audits it against every other effective library
artifact using each factor's declared signal_artifact path.
"""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2030-08-21'); SELF='miner_2_contrarian_range_position_pressure_10x20obs'
def load(a):
 x=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return x[['close','high','low']].astype(float)
X={a:load(a) for a in A}; C=pd.DataFrame({a:X[a].close for a in A}).sort_index()
clv=pd.DataFrame({a:(2*X[a].close-X[a].high-X[a].low)/(X[a].high-X[a].low).replace(0,np.nan) for a in A})
rr=pd.DataFrame({a:(X[a].high-X[a].low)/X[a].close.replace(0,np.nan) for a in A})
F=-(clv.rolling(10,min_periods=8).mean()/rr.rolling(20,min_periods=15).mean().replace(0,np.nan))
F=F.sub(F.median(axis=1),axis=0).loc[:END]
def calc(h):
 fut=(C.shift(-h)/C-1).reindex(F.index); vals=[]; ns=[]
 for d in F.index:
  q=pd.concat([F.loc[d].rename('f'),fut.loc[d].rename('r')],axis=1).dropna()
  if len(q)>=8:
   z=spearmanr(q.f,q.r).statistic
   if np.isfinite(z): vals.append((d,float(z))); ns.append(len(q))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
ALL={}
for h in (1,5,10,20):
 s,ALL[h]=calc(h); print('HORIZON',h,json.dumps(ALL[h],sort_keys=True))
s,_=calc(5)
for lab,m in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year>=2027)]:
 z=s[m]; print('REGIME_5D',lab,'dates',len(z),'IC',float(z.mean()),'ICIR',float(z.mean()/z.std(ddof=1)),'hit',float((z>0).mean()))
st=[]
for i in range(1,len(F)):
 q=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(q)>=8: st.append(float(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic))
mx=0.;who=None;ev={}; audited=0
for p in glob.glob('factors/*.json'):
 if p.endswith('.bak') or '_deprecated' in p: continue
 try: d=json.load(open(p))
 except: continue
 fid=d.get('factor_id');
 if fid==SELF or d.get('validation',{}).get('status')!='EFFECTIVE': continue
 audited+=1; artifact=d.get('signal_artifact')
 if not artifact or not os.path.exists(artifact): mx=np.inf;ev[fid]={'rho':None,'common_signal_cells':0,'file':artifact};continue
 try:
  L=pd.read_pickle(artifact).reindex(index=F.index,columns=A); q=pd.concat([F.stack().rename('f'),L.stack().rename('l')],axis=1).dropna(); rho=float(spearmanr(q.f,q.l).statistic) if len(q)>=8 else np.nan
 except Exception: q=pd.DataFrame();rho=np.nan
 ev[fid]={'rho':rho if np.isfinite(rho) else None,'common_signal_cells':len(q),'file':artifact}
 if not np.isfinite(rho): mx=np.inf
 elif abs(rho)>mx: mx=abs(rho);who=fid
print('FACTOR',SELF)
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in ALL.items()},sort_keys=True))
print('MAX_ABS_LIBRARY_CORRELATION',mx,'FACTOR',who,'AUDITED',audited,'EVIDENCE',json.dumps(ev,sort_keys=True))
F.to_pickle('scripts/miner_2_20300822_contrarian_range_position_pressure_10x20obs_signal.pkl')
