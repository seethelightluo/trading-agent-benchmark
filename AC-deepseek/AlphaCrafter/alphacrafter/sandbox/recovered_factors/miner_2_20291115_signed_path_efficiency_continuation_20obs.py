"""One candidate: signed 20-day return-path efficiency continuation.
The signal is net 20d return divided by accumulated absolute daily moves:
a persistent directional path receives a large signed score, while a choppy
move with same endpoint is discounted.  It tests continuation conditional on
path quality across the 15-asset benchmark.
"""
import glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-11-14')
def load(a):
 d=pd.read_csv(f'../persistent/stock_data/{a}.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,'close'].astype(float)
P=pd.DataFrame({a:load(a) for a in A}).sort_index()
r=P.pct_change()
# All terms end at t; signal is defined only after a full 20-observation path.
F=(P.pct_change(20).div(r.abs().rolling(20,min_periods=20).sum())).replace([np.inf,-np.inf],np.nan)
def ev(h):
 fw=P.shift(-h).div(P).sub(1);v=[];ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:
   v.append((dt,float(spearmanr(z.f,z.r).statistic)));ns.append(len(z))
 s=pd.Series(dict(v)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments_per_ic_date':float(np.mean(ns))}
S={}
for h in (1,5,10,20):
 S[h],m=ev(h);print('HORIZON',h,json.dumps(m,sort_keys=True))
for name,mask in [('2020_2022',S[5].index.year<=2022),('2023_2024',S[5].index.year.isin([2023,2024])),('2025_2026',S[5].index.year.isin([2025,2026])),('2027_2028',S[5].index.year.isin([2027,2028])),('2029_ytd',S[5].index.year==2029)]:
 s=S[5][mask];print('REGIME_5',name,json.dumps({'dates':len(s),'ic':float(s.mean()) if len(s) else None,'icir':float(s.mean()/s.std(ddof=1)) if len(s)>1 else None,'hit':float((s>0).mean()) if len(s) else None}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('PANEL',json.dumps({'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_valid_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'implied_turnover':float(1-np.mean(st))}))
complete=True;mx=-1.;who=None;n=0;available=0
for p in glob.glob('factors/*.json'):
 if p.endswith('.bak'):continue
 d=json.load(open(p))
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 n+=1;fid=d['factor_id'];hit=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not hit: print('LIB_MISSING',fid);complete=False;continue
 available+=1;G=pd.read_pickle(sorted(hit)[-1]);x,y=F.align(G,join='inner',axis=0);z=pd.DataFrame({'x':x.stack(),'y':y.stack()}).dropna();rho=float(spearmanr(z.x,z.y).statistic) if len(z)>2 else np.nan
 print('LIB_CORR',fid,len(z),rho)
 if not np.isfinite(rho):complete=False
 elif abs(rho)>mx:mx=abs(rho);who=fid
print('AUDIT',json.dumps({'effective_factor_count':n,'available_comparisons':available,'complete':complete,'max_abs_library_correlation':mx if complete else None,'observed_max_abs_library_correlation':mx,'most_correlated':who}))
F.to_pickle('scripts/miner_2_20291115_signed_path_efficiency_continuation_20obs_signal.pkl')
