"""Miner 2: price-only loss-frequency imbalance over 20 observations."""
import os,glob,json,warnings
import numpy as np,pandas as pd
from scipy.stats import spearmanr,ConstantInputWarning
warnings.filterwarnings('ignore',category=ConstantInputWarning)
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2034-11-22'); FID='miner_2_loss_frequency_imbalance_20obs'
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d['close'].astype(float)
C=pd.DataFrame({a:load(a) for a in A}).loc[:END]; r=C.pct_change()
# Net share of down versus up sessions: +1 if all 20 sessions declined, -1 if all rose.
F=((r<0).rolling(20,min_periods=16).sum()-(r>0).rolling(20,min_periods=16).sum()).div((r!=0).rolling(20,min_periods=16).sum()).replace([np.inf,-np.inf],np.nan)
print('INPUT_NONMISSING_CLOSE',float(C.notna().mean().mean()))
def ev(h):
 y=C.shift(-h).div(C).sub(1); out=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((dt,float(q)));ns.append(len(z))
 s=pd.Series(dict(out),dtype=float);sd=s.std(ddof=1)
 return s,{'paper_ic':float(s.mean()) if len(s) else None,'paper_icir':float(s.mean()/sd) if len(s)>1 and sd else None,'hit_ratio':float((s>0).mean()) if len(s) else None,'standard_error':float(sd/np.sqrt(len(s))) if len(s)>1 else None,'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns)) if ns else 0}
MET={}
for h in (1,5,10,20):
 s,MET[h]=ev(h);print('HORIZON',h,json.dumps(MET[h],sort_keys=True))
s,_=ev(1);s.index=pd.to_datetime(s.index)
for n,yy in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('2032_2034',[2032,2033,2034])]:
 x=s[s.index.year.isin(yy)];sd=x.std(ddof=1);print('REGIME_1D',n,'dates',len(x),'ic',float(x.mean()) if len(x) else None,'icir',float(x.mean()/sd) if len(x)>1 and sd else None,'hit',float((x>0).mean()) if len(x) else None)
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
 except Exception:pass
E={};complete=True;mx=0.;most=None
for fid in active:
 key=fid
 for pre in ('miner_1_','miner_2_','miner_3_'):key=key.replace(pre,'')
 ps=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not ps:E[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 p=max(ps,key=os.path.getmtime)
 try:
  Q=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('candidate'),Q.stack().rename('library')],axis=1).dropna();q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic if len(z)>=8 else np.nan
 except Exception:z=pd.DataFrame();q=np.nan
 E[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('FACTOR',FID);print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability',float(np.mean(st)),'turnover',float(1-np.mean(st)))
print('DECAY',json.dumps(MET,sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(E,sort_keys=True));F.to_pickle('scripts/miner_2_20341123_loss_frequency_imbalance_20obs_signal.pkl')
