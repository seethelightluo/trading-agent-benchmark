"""Candidate: volume-amplified short-term return exhaustion, cutoff 2028-12-14.
Score = -(5-day return) * log(mean volume last 5d / mean volume last 20d).
A large directional move accompanied by unusually high participation is treated as
cross-asset exhaustion, so the sign reverses the recent return. Inputs at t only.
"""
import glob,json,os
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2028-12-14')
def field(a,x):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.loc[:END,x].astype(float)
P=pd.DataFrame({a:field(a,'close') for a in A}).sort_index(); V=pd.DataFrame({a:field(a,'volume') for a in A}).sort_index()
F=-(P/P.shift(5)-1)*np.log(V.rolling(5,min_periods=4).mean()/V.rolling(20,min_periods=15).mean().replace(0,np.nan))
def calc(h):
 fw=P.shift(-h)/P-1; out=[]; ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),fw.loc[d].rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
  if len(z)>=8 and z.f.nunique()>1 and z.r.nunique()>1: out.append((d,float(spearmanr(z.f,z.r).statistic)));ns.append(len(z))
 s=pd.Series(dict(out),dtype=float); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for h in [1,5,10,20]:
 s,m=calc(h); print('HORIZON',h,json.dumps(m,sort_keys=True))
 if h==5:
  for n,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2025',s.index.year.isin([2024,2025])),('2026_2028',s.index.year>=2026)]:
   q=s[mask]; print('REGIME_5D',n,json.dumps({'dates':len(q),'ic':float(q.mean()) if len(q) else None,'icir':float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,'hit':float((q>0).mean()) if len(q) else None}))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('PANEL',json.dumps({'signal_dates':len(F),'coverage':float(F.replace([np.inf,-np.inf],np.nan).notna().mean().mean()),'mean_names':float(F.replace([np.inf,-np.inf],np.nan).notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'implied_turnover':float(1-np.mean(st))}))
F=F.replace([np.inf,-np.inf],np.nan); F.to_pickle('scripts/miner_2_20281214_volume_amplified_return_exhaustion_5v20obs_signal.pkl')
alias={}
# Use names from every effective factor file and locate their established signal artifact by factor-id substring.
for p in glob.glob('factors/*.json'):
 if p.endswith('.bak'):continue
 d=json.load(open(p)); fid=d.get('factor_id','')
 if d.get('validation',{}).get('status')=='EFFECTIVE':
  matches=glob.glob('scripts/*'+fid+'*signal.pkl')
  # Older artifacts sometimes omit the miner prefix; retain known matching via final descriptive id.
  if not matches: matches=glob.glob('scripts/*'+fid.replace('miner_2_','').replace('miner_1_','').replace('miner_3_','')+'*signal.pkl')
  alias[fid]=matches[-1] if matches else None
rows=[];missing=[]
for fid,fn in alias.items():
 if not fn: missing.append(fid);continue
 L=pd.read_pickle(fn); x,y=F.align(L,join='inner',axis=0); z=pd.DataFrame({'x':x.stack(),'y':y.stack()}).dropna()
 if len(z)<3: missing.append(fid);continue
 rho=float(spearmanr(z.x,z.y).statistic);rows.append((fid,len(z),rho));print('LIBRARY_CORR',fid,len(z),rho)
print('LIBRARY_MISSING',json.dumps(missing))
if not missing:
 b=max(rows,key=lambda x:abs(x[2]));print('LIBRARY_MAX',json.dumps({'factor':b[0],'cells':b[1],'rho':b[2],'max_abs_library_correlation':abs(b[2]),'audited_factors':len(rows)}))
