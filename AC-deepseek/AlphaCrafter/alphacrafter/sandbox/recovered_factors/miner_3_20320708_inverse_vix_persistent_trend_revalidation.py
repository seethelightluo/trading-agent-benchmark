"""Miner_3 scheduled revalidation of inverse VIX persistent trend; visible data through 2032-07-07."""
import os,glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2032-07-07'); FID='miner_3_inverse_vix_persistent_trend_exposure_10v40v50obs'
def close(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index().loc[:END]; vix=close('../persistent/index_data/VIX.csv').reindex(C.index).ffill(); r=np.log(C).diff(); vr=np.log(vix).diff()
beta=r.rolling(50,min_periods=35).cov(vr).div(vr.rolling(50,min_periods=35).var(),axis=0); imp=vr.rolling(10,min_periods=10).sum(); F=beta.mul(imp.where(imp*vr.rolling(40,min_periods=40).sum()>0),axis=0)
def calc(h):
 y=C.shift(-h)/C-1; rows=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q): rows.append((d,float(q)));ns.append(len(z))
 ic=pd.Series(dict(rows)); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(ns))}
M={}
for h in (1,5,10,20): _,M[h]=calc(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic,_=calc(10)
for lab,yrs in [('2024_2026',[2024,2025,2026]),('2027_2030',[2027,2028,2029,2030]),('2031',[2031]),('2032_YTD',[2032])]:
 x=ic[ic.index.year.isin(yrs)];print('REGIME',lab,'dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)),'hit',float((x>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):st.append(q)
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except: pass
E={};mx=0.;most=None;complete=True
for fid in active:
 if fid==FID:continue
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_',''); key='state_gated_inverse_volatility_expansion_10v60obs' if fid=='miner_1_state_gated_volatility_expansion_10v60obs' else key
 ps=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not ps:E[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 p=max(ps,key=os.path.getmtime)
 try:
  lib=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('candidate'),lib.stack().rename('library')],axis=1).dropna();q=spearmanr(z.candidate,z.library).statistic if len(z)>=8 else np.nan
 except: z=pd.DataFrame();q=np.nan
 E[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(float(q));most=fid
print('PANEL',F.index.min().date(),END.date(),len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability',float(np.mean(st)),'turnover',float(1-np.mean(st)))
print('MAXCORR',mx,'MOST',most,'COMPLETE',complete,'COMPARED',len(active)-1);print('EVIDENCE',json.dumps(E,sort_keys=True));F.to_pickle('scripts/miner_3_20320708_inverse_vix_persistent_trend_revalidation_signal.pkl')
