"""Miner_3 revalidation: elevated-VIX conditional DXY impulse; visible through 2032-04-28."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2032-04-28')
def close(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index().loc[:END]
dxy=close('../persistent/index_data/DXY.csv').reindex(C.index).ffill(); vix=close('../persistent/index_data/VIX.csv').reindex(C.index).ffill()
r=np.log(C).diff(); dr=np.log(dxy).diff(); beta=r.rolling(40,min_periods=30).cov(dr).div(dr.rolling(40,min_periods=30).var(),axis=0)
imp=dr.rolling(5,min_periods=5).sum(); state=(vix>vix.rolling(20,min_periods=15).mean()).astype(float)
F=beta.mul(-imp*state,axis=0).where(state.astype(bool),np.nan)
def calc(h):
 y=C.shift(-h).div(C).sub(1); out=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((d,float(q)));ns.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments':float(np.mean(ns))}
R={}
for h in (1,5,10,20):
 x,R[h]=calc(h); print('HORIZON',h,json.dumps(R[h],sort_keys=True))
x,_=calc(20)
for name,yr in [('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031',[2031]),('2032_YTD',[2032])]:
 q=x[x.index.year.isin(yr)]; print('REGIME',name,'dates',len(q),'IC',float(q.mean()),'ICIR',float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,'hit',float((q>0).mean()))
st=[]
for k in range(1,len(F)):
 z=pd.concat([F.iloc[k-1],F.iloc[k]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):st.append(q)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE': active.append(j['factor_id'])
 except: pass
mx=0.;most=None;ev={};complete=True
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_',''); ps=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not ps: ev[fid]={'rho':None,'cells':0};complete=False;continue
 p=max(ps,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('a'),L.stack().rename('b')],axis=1).dropna(); q=spearmanr(z.a,z.b).statistic if len(z)>=8 else np.nan
 except: z=pd.DataFrame();q=np.nan
 ev[fid]={'rho':float(q) if np.isfinite(q) else None,'cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('PANEL',F.index.min().date(),END.date(),'dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability',float(np.mean(st)),'turnover',float(1-np.mean(st)))
print('LIBRARY','max_abs',mx,'most',most,'complete',complete,json.dumps(ev,sort_keys=True))
F.to_pickle('scripts/miner_3_20320429_dxy_vix_revalidation_5v40v20obs_signal.pkl')
