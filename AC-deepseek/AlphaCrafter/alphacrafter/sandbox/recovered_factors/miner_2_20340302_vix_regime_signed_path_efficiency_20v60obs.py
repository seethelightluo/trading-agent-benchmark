"""Miner 2: VIX-regime signed 20-day return-path efficiency, close-only candidate."""
import os,glob,json,warnings
import numpy as np,pandas as pd
from scipy.stats import spearmanr,ConstantInputWarning
warnings.filterwarnings('ignore',category=ConstantInputWarning)
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2034-03-01'); FID='miner_2_vix_regime_signed_path_efficiency_20v60obs'
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
def ix(a): return pd.read_csv('../persistent/index_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
C=pd.DataFrame({a:load(a) for a in A}).loc[:END]; R=C.pct_change()
v=ix('VIX').reindex(C.index).ffill(); stress=v>v.rolling(60,min_periods=45).median()
# Efficiency distinguishes directional paths from choppy paths. Retain efficiency in stress, invert it in calm markets.
eff=C.pct_change(20).abs().div(R.abs().rolling(20,min_periods=16).sum())
F=eff.mul(np.where(stress,1.,-1.),axis=0)
def ev(h):
 y=C.shift(-h).div(C).sub(1);out=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((dt,float(q)));ns.append(len(z))
 s=pd.Series(dict(out),dtype=float);d=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/d) if d else None,'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(d/np.sqrt(len(s))) if len(s) else None,'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns)) if ns else 0}
MET={}
for h in (1,5,10,20):
 s,MET[h]=ev(h);print('HORIZON',h,json.dumps(MET[h],sort_keys=True))
for h in (1,5,10,20):
 s,_=ev(h)
 for n,yy in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('recent_2032_2034',[2032,2033,2034])]:
  x=s[s.index.year.isin(yy)];d=x.std(ddof=1);print('REGIME',h,n,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/d) if len(x)>1 and d else None,'hit',float((x>0).mean()) if len(x) else None)
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
 except: pass
E={};complete=True;mx=0.;most=None
for fid in active:
 key=fid
 for pre in ('miner_1_','miner_2_','miner_3_'):key=key.replace(pre,'')
 paths=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not paths:E[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 p=max(paths,key=os.path.getmtime)
 try:
  Q=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('candidate'),Q.stack().rename('library')],axis=1).dropna();q=spearmanr(z.candidate,z.library).statistic if len(z)>=8 else np.nan
 except: z=pd.DataFrame();q=np.nan
 E[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('FACTOR',FID);print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'stress_frequency',float(stress.mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in MET.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(E,sort_keys=True))
F.to_pickle('scripts/miner_2_20340302_vix_regime_signed_path_efficiency_20v60obs_signal.pkl')
