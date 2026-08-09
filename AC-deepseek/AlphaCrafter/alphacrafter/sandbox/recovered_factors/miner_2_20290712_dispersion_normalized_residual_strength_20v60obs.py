"""One candidate: dispersion-normalized residual relative strength.
Estimate each asset's 60-session beta to the equal-weighted cross-asset daily
return. Score is its 20-session return less beta times the corresponding
cross-asset return, but is emitted only while recent (5d) cross-sectional
return dispersion is below its preceding 20d level. This tests whether
idiosyncratic medium-term strength is more informative after shocks normalize.
"""
import glob,json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2029-07-11')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,'close'].astype(float)
P=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=P.pct_change(); M=R.mean(axis=1)
# Rolling beta is estimated exclusively from completed trailing observations.
beta=R.rolling(60,min_periods=45).cov(M).div(M.rolling(60,min_periods=45).var(),axis=0)
r20=P.pct_change(20); m20=(1+M).rolling(20,min_periods=15).apply(np.prod,raw=True)-1
resid=r20-beta.mul(m20,axis=0)
disp5=R.std(axis=1).rolling(5,min_periods=5).mean(); disp20=R.std(axis=1).rolling(20,min_periods=15).mean()
state=disp5 < disp20.shift(5) # known at t, requires an observable dispersion contraction
F=resid.where(state, np.nan).replace([np.inf,-np.inf],np.nan)
def calc(h):
 fw=P.shift(-h).div(P).sub(1); out=[]; ns=[]
 for dt in F.index:
  z=pd.concat([F.loc[dt].rename('f'),fw.loc[dt].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1:
   out.append((dt,float(spearmanr(z.f,z.r).statistic)));ns.append(len(z))
 s=pd.Series(dict(out),dtype=float); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for h in [1,5,10,20]:
 s,m=calc(h);print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==1:
  for lab,mask in [('2020_2022',s.index.year<=2022),('2023_2025',s.index.year.isin([2023,2024,2025])),('2026_2028',s.index.year.isin([2026,2027,2028])),('2029_ytd',s.index.year==2029)]:
   q=s[mask];print('REGIME',lab,json.dumps({'ic':float(q.mean()),'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,'hit_ratio':float((q>0).mean()),'ic_dates':len(q)}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('PANEL',json.dumps({'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_valid_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'implied_turnover':float(1-np.mean(st)),'state_dates':int(state.sum())}))
complete=True;mx=-1.;who=None;effective=[]
for path in glob.glob('factors/*.json'):
 if path.endswith('.bak'):continue
 d=json.load(open(path))
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 fid=d['factor_id'];effective.append(fid);hits=glob.glob('scripts/*'+fid+'*signal.pkl')
 if not hits:print('LIB_MISSING',fid);complete=False;continue
 G=pd.read_pickle(sorted(hits)[-1]);x,y=F.align(G,join='inner',axis=0);z=pd.DataFrame({'x':x.stack(),'y':y.stack()}).dropna()
 rho=float(spearmanr(z.x,z.y).statistic) if len(z)>2 else np.nan;print('LIB_CORR',fid,len(z),rho)
 if not np.isfinite(rho):complete=False
 elif abs(rho)>mx:mx=abs(rho);who=fid
print('AUDIT',json.dumps({'effective_factor_count':len(effective),'complete':complete,'max_abs_library_correlation':mx if complete else None,'observed_max_abs_library_correlation':mx,'most_correlated':who}))
F.to_pickle('scripts/miner_2_20290712_dispersion_normalized_residual_strength_20v60obs_signal.pkl')
