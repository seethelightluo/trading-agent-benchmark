"""Miner 2 2034-10-26: down-volume participation asymmetry, one candidate."""
import os,glob,json,warnings
import numpy as np,pandas as pd
from scipy.stats import spearmanr,ConstantInputWarning
warnings.filterwarnings('ignore',category=ConstantInputWarning)
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];END=pd.Timestamp('2034-10-25');FID='miner_2_down_volume_participation_asymmetry_10v60obs'
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index();return d.close.astype(float),d.volume.astype(float)
D={a:load(a) for a in A};C=pd.DataFrame({a:D[a][0] for a in A}).loc[:END];V=pd.DataFrame({a:D[a][1] for a in A}).loc[:END]
r=C.pct_change();base=V.rolling(60,min_periods=40).mean(); dn=(r<0);up=~dn
# mean volume conditional on down/up sessions, each requiring at least 3 occurrences in ten sessions.
dn_n=dn.rolling(10).sum();up_n=up.rolling(10).sum();dn_v=V.where(dn).rolling(10).sum().div(dn_n.where(dn_n>=3));up_v=V.where(up).rolling(10).sum().div(up_n.where(up_n>=3))
F=np.log(dn_v.div(up_v)).replace([np.inf,-np.inf],np.nan).clip(-3,3)
def ev(h):
 y=C.shift(-h).div(C).sub(1);out=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((dt,float(q)));ns.append(len(z))
 s=pd.Series(dict(out),dtype=float);sd=s.std(ddof=1)
 return s,{'paper_ic':float(s.mean()) if len(s) else None,'paper_icir':float(s.mean()/sd) if sd else None,'hit_ratio':float((s>0).mean()) if len(s) else None,'standard_error':float(sd/np.sqrt(len(s))) if len(s) else None,'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns)) if ns else 0}
MET={}
for h in (1,5,10,20):s,MET[h]=ev(h);print('HORIZON',h,json.dumps(MET[h],sort_keys=True))
s,_=ev(1);s.index=pd.to_datetime(s.index)
for n,ys in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('2032_2034',[2032,2033,2034])]:
 x=s[s.index.year.isin(ys)];sd=x.std(ddof=1);print('REGIME_1D',n,len(x),float(x.mean()) if len(x) else None,float(x.mean()/sd) if len(x)>1 and sd else None,float((x>0).mean()) if len(x) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except:pass
E={};complete=True;mx=0.;most=None
for fid in active:
 key=fid
 for pre in ('miner_1_','miner_2_','miner_3_'):key=key.replace(pre,'')
 paths=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not paths:E[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 p=max(paths,key=os.path.getmtime)
 try:
  Q=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('candidate'),Q.stack().rename('library')],axis=1).dropna();q=spearmanr(z.candidate,z.library).statistic if len(z)>=8 else np.nan
 except:z=pd.DataFrame();q=np.nan
 E[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('FACTOR',FID);print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability',float(np.nanmean(st)),'turnover',float(1-np.nanmean(st)))
print('DECAY',json.dumps(MET,sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(E,sort_keys=True));F.to_pickle('scripts/miner_2_20341026_down_volume_participation_asymmetry_10v60obs_signal.pkl')
