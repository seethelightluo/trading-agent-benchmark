"""Miner_3: elevated-VIX shock transmission exposure, visible through 2032-04-14."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2032-04-14')
def close(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index().loc[:END]
vix=close('../persistent/index_data/VIX.csv').reindex(C.index).ffill();r=np.log(C).diff();vr=np.log(vix).diff()
# A VIX shock should be informative only when VIX is already elevated versus its trailing state.
beta=r.rolling(40,min_periods=30).cov(vr).div(vr.rolling(40,min_periods=30).var(),axis=0)
shock=vr.rolling(5,min_periods=5).sum(); elevated=(vix>vix.rolling(60,min_periods=45).median()).astype(float)
F=beta.mul(-shock*(shock>0).astype(float)*elevated,axis=0)
def calc(h):
 y=C.shift(-h).div(C).sub(1);out=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((d,float(q)));ns.append(len(z))
 x=pd.Series(dict(out));sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments':float(np.mean(ns))}
M={}
for h in (1,5,10,20):
 x,M[h]=calc(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic,_=calc(20)
for label,yrs in [('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031',[2031]),('2032_YTD',[2032])]:
 x=ic[ic.index.year.isin(yrs)];print('REGIME',label,'dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()))
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
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except:pass
mx=0.;most=None;ev={};ok=True
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','');ps=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not ps:ev[fid]={'rho':None,'cells':0};ok=False;continue
 p=max(ps,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('a'),L.stack().rename('b')],axis=1).dropna();q=spearmanr(z.a,z.b).statistic if len(z)>=8 else np.nan
 except: z=pd.DataFrame();q=np.nan
 ev[fid]={'rho':float(q) if np.isfinite(q) else None,'cells':len(z),'file':p}
 if not np.isfinite(q):ok=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('PANEL',F.index.min().date(),END.date(),'dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability',float(np.mean(st)),'turnover',float(1-np.mean(st)))
print('LIBRARY','max_abs',mx,'most',most,'complete',ok,json.dumps(ev,sort_keys=True))
F.to_pickle('scripts/miner_3_20320415_elevated_vix_shock_transmission_5v40v60obs_signal.pkl')
