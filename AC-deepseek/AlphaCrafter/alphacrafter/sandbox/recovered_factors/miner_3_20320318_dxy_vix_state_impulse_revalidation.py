"""Miner_3 scheduled revalidation: Elevated-VIX Conditional DXY Impulse Exposure through 2032-03-17."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2032-03-17')
def close(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index().loc[:END]
dxy=close('../persistent/index_data/DXY.csv').reindex(C.index).ffill(); vix=close('../persistent/index_data/VIX.csv').reindex(C.index).ffill(); r=np.log(C).diff(); dr=np.log(dxy).diff()
beta=r.rolling(40,min_periods=30).cov(dr).div(dr.rolling(40,min_periods=30).var(),axis=0); state=(vix>vix.rolling(20,min_periods=15).mean()).astype(float); F=beta.mul(-dr.rolling(5,min_periods=5).sum()*state,axis=0)
def calc(h):
 y=C.shift(-h).div(C).sub(1); vals=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): vals.append((d,float(q)));ns.append(len(z))
 ic=pd.Series(dict(vals)); sd=ic.std(ddof=1); return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(ns))}
M={}
for h in (1,5,10,20): _,M[h]=calc(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic,_=calc(20)
for lab,yrs in [('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031',[2031]),('2032_YTD',[2032])]:
 x=ic[ic.index.year.isin(yrs)]; print('REGIME',lab,'dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):st.append(float(q))
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except:pass
mx=0.;most=None;complete=True;evidence={}
for fid in active:
 if fid=='miner_3_dxy_vix_state_impulse_exposure_5v40v20obs':continue
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_',''); ps=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not ps: evidence[fid]={'rho':None,'cells':0};complete=False;continue
 p=max(ps,key=os.path.getmtime)
 try:
  lib=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('a'),lib.stack().rename('b')],axis=1).dropna();q=spearmanr(z.a,z.b).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 evidence[fid]={'rho':float(q) if np.isfinite(q) else None,'cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('PANEL',F.index.min().date(),END.date(),'dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability',float(np.mean(st)),'turnover',float(1-np.mean(st)))
print('LIBRARY', 'max_abs',mx,'most',most,'complete',complete,json.dumps(evidence,sort_keys=True));F.to_pickle('scripts/miner_3_20320318_dxy_vix_state_impulse_revalidation_signal.pkl')
