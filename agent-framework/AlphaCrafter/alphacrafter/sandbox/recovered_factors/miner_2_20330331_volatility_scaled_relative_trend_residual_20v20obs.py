"""Miner_2 single-idea research: cross-asset relative trend residual, cutoff 2033-03-30.
Assets whose 20d return exceeds the contemporaneous equal-universe median, scaled by
own trailing volatility, may retain leadership after removing common macro drift.
"""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2033-03-30'); FID='miner_2_volatility_scaled_relative_trend_residual_20v20obs'
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.close.astype(float)
C=pd.DataFrame({a:load(a) for a in A}).loc[:END]; r=np.log(C).diff()
raw=C.pct_change(20); common=raw.median(axis=1); vol=r.rolling(20,min_periods=15).std()
F=raw.sub(common,axis=0).div(vol).replace([np.inf,-np.inf],np.nan)
def ev(h):
 y=C.shift(-h).div(C).sub(1); out=[]; n=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((d,float(q)));n.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments':float(np.mean(n))}
M={}
for h in [1,5,10,20]:
 x,M[h]=ev(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic,_=ev(5)
for lab,yrs in [('2020_2022',[2020,2021,2022]),('2023_2025',[2023,2024,2025]),('2026_2028',[2026,2027,2028]),('2029_2031',[2029,2030,2031]),('2032_2033',[2032,2033])]:
 x=ic[ic.index.year.isin(yrs)]; sd=x.std(ddof=1);print('REGIME_5D',lab,'dates',len(x),'IC',float(x.mean()) if len(x) else None,'ICIR',float(x.mean()/sd) if len(x)>1 and sd else None,'hit',float((x>0).mean()) if len(x) else None)
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
 except:pass
complete=True;mx=0.;most=None;e={}
for fid in active:
 key=fid
 for pref in ('miner_1_','miner_2_','miner_3_'):key=key.replace(pref,'')
 ps=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not ps:e[fid]={'rho':None,'common_signal_cells':0};complete=False;continue
 p=max(ps,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('a'),L.stack().rename('b')],axis=1).dropna();q=spearmanr(z.a,z.b).statistic if len(z)>=8 else np.nan
 except Exception:q=np.nan;z=pd.DataFrame()
 e[fid]={'rho':float(q) if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);most=fid
print('FACTOR',FID);print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'mean_rank_stability_1d',float(np.mean(st)),'implied_rank_turnover',float(1-np.mean(st)))
print('DECAY',json.dumps({str(k):v for k,v in M.items()}));print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'COMPLETE',complete,'EVIDENCE',json.dumps(e,sort_keys=True))
F.to_pickle('scripts/miner_2_20330331_volatility_scaled_relative_trend_residual_20v20obs_signal.pkl')
