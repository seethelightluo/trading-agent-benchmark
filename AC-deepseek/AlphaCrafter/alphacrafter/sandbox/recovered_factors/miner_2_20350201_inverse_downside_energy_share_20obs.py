"""Miner_2 single-factor test: inverse downside return-energy share, cutoff 2035-01-31."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2035-01-31')
def load(s): return pd.read_csv(f'../persistent/stock_data/{s}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
C=pd.DataFrame({a:load(a) for a in A}).loc[:END]; r=np.log(C).diff()
# Assets whose recent path contains a smaller share of negative return energy are comparatively resilient.
down=r.clip(upper=0).pow(2).rolling(20,min_periods=15).sum(); total=r.pow(2).rolling(20,min_periods=15).sum()
F=-(down/total.replace(0,np.nan))
def met(h):
 y=C.shift(-h).div(C)-1; out=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): out.append((d,q));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments':float(np.mean(ns))}
R={}
for h in [1,5,10,20]:
 x,m=met(h); R[h]=m; print('HORIZON',h,json.dumps(m,sort_keys=True))
x,_=met(1)
for name,yrs in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('2032_2035',[2032,2033,2034,2035])]:
 z=x[x.index.year.isin(yrs)]; print('REGIME',name,'DATES',len(z),'IC',float(z.mean()) if len(z) else None,'ICIR',float(z.mean()/z.std(ddof=1)) if len(z)>1 else None,'HIT',float((z>0).mean()) if len(z) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append((j['factor_id'],p))
 except: pass
files=glob.glob('scripts/*_signal.pkl'); ev={};mx=0.;most=None;complete=True
for fid,p in active:
 qf=[q for q in files if fid in os.path.basename(q)]
 if not qf:qf=[q for q in files if os.path.basename(p).replace('.json','').split('_',3)[-1] in os.path.basename(q)]
 if not qf:ev[fid]=None;complete=False;continue
 try:
  L=pd.read_pickle(max(qf,key=os.path.getmtime)).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('f'),L.stack().rename('l')],axis=1).dropna();q=spearmanr(z.f,z.l).statistic if len(z)>=8 and z.f.nunique()>1 and z.l.nunique()>1 else np.nan
 except: q=np.nan
 ev[fid]=float(q) if np.isfinite(q) else None
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('PANEL_DATES',len(F),'UNIVERSE',len(A),'COVERAGE',float(F.notna().mean().mean()),'MEAN_NAMES',float(F.notna().sum(axis=1).mean()),'STABILITY',float(np.nanmean(st)),'TURNOVER',float(1-np.nanmean(st)))
print('MAXCORR',mx,'MOST',most,'COMPLETE',complete,'COMPARED',len(active));print('EVIDENCE',json.dumps(ev,sort_keys=True))
F.to_pickle('scripts/miner_2_20350201_inverse_downside_energy_share_20obs_signal.pkl')
