"""Miner 2: inverse recent-to-baseline downside semivolatility ratio, evaluated without future data."""
import os, glob, json, warnings
import numpy as np, pandas as pd
from scipy.stats import spearmanr, ConstantInputWarning
warnings.filterwarnings('ignore', category=ConstantInputWarning)
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2034-03-15'); FID='miner_2_inverse_downside_semivol_compression_5v20obs'
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.close.astype(float)
C=pd.DataFrame({a:load(a) for a in A}).loc[:END]; R=C.pct_change()
# The pre-specified score rewards assets whose recent adverse-return variability has compressed versus their own 20-observation baseline.
down=R.clip(upper=0)
d5=np.sqrt((down**2).rolling(5,min_periods=4).mean())
d20=np.sqrt((down**2).rolling(20,min_periods=16).mean())
F=-(d5/(d20.replace(0,np.nan)))
def ev(h):
 y=C.shift(-h).div(C).sub(1); out=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): out.append((dt,float(q))); ns.append(len(z))
 s=pd.Series(dict(out),dtype=float); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd) if sd else None,'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))) if len(s) else None,'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns)) if ns else 0}
MET={}
for h in (1,5,10,20):
 s,MET[h]=ev(h); print('HORIZON',h,json.dumps(MET[h],sort_keys=True))
s,_=ev(5)
for name,yrs in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('recent_2032_2034',[2032,2033,2034])]:
 x=s[s.index.year.isin(yrs)]; sd=x.std(ddof=1)
 print('REGIME_5D',name,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/sd) if len(x)>1 and sd else None,'hit',float((x>0).mean()) if len(x) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):st.append(q)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except Exception: pass
E={}; complete=True; mx=0.; most=None
for fid in active:
 key=fid
 for pre in ('miner_1_','miner_2_','miner_3_'): key=key.replace(pre,'')
 paths=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not paths: E[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 p=max(paths,key=os.path.getmtime)
 try:
  Q=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('candidate'),Q.stack().rename('library')],axis=1).dropna(); q=spearmanr(z.candidate,z.library).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 E[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('FACTOR',FID); print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)) if st else None,'implied_rank_turnover',float(1-np.mean(st)) if st else None)
print('DECAY',json.dumps({str(k):v for k,v in MET.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(E,sort_keys=True))
F.to_pickle('scripts/miner_2_20340316_inverse_downside_semivol_compression_5v20obs_signal.pkl')
