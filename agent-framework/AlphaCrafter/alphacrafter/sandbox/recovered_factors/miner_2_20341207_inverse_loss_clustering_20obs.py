"""Miner 2: loss-clustering shape, 20 observations; candidate validation."""
import os,glob,json,warnings
import numpy as np,pandas as pd
from scipy.stats import spearmanr,ConstantInputWarning
warnings.filterwarnings('ignore',category=ConstantInputWarning)
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2034-12-06'); FID='miner_2_inverse_loss_clustering_20obs'
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
C=pd.DataFrame({a:load(a) for a in A}).loc[:END]; R=C.pct_change()
def cluster(x):
 x=x[np.isfinite(x)]
 if len(x)<16:return np.nan
 b=(x<0).astype(int); neg=b.sum()
 if neg==0:return 0.
 # fraction of negative sessions embodied in the longest contiguous loss run
 runs=np.diff(np.r_[0,np.where(b==0,0,1),0]); starts=np.where(runs==1)[0]; ends=np.where(runs==-1)[0]
 return -max(ends-starts)/neg
F=R.rolling(20,min_periods=16).apply(cluster,raw=True)
def ev(h):
 y=C.shift(-h).div(C).sub(1); out=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((dt,float(q)));ns.append(len(z))
 s=pd.Series(dict(out),dtype=float); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd) if sd else None,'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
MET={}
for h in (1,5,10,20): s,MET[h]=ev(h);print('HORIZON',h,json.dumps(MET[h],sort_keys=True))
s,_=ev(1)
for n,ys in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('2032_2034',[2032,2033,2034])]:
 x=s[s.index.year.isin(ys)]; print('REGIME_1D',n,'dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)),'hit',float((x>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
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
 paths=glob.glob('scripts/*'+key+'*_signal.pkl')
 if not paths:E[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 p=max(paths,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('f'),L.stack().rename('l')],axis=1).dropna();q=spearmanr(z.f,z.l).statistic if len(z)>=8 else np.nan
 except: z=pd.DataFrame();q=np.nan
 E[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);most=fid
print('FACTOR',FID);print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.nanmean(st)),'implied_rank_turnover',float(1-np.nanmean(st)))
print('DECAY',json.dumps({str(k):v for k,v in MET.items()},sort_keys=True));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(E,sort_keys=True))
F.to_pickle('scripts/miner_2_20341207_inverse_loss_clustering_20obs_signal.pkl')
