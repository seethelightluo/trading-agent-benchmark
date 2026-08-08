"""One candidate: cross-asset liquidity acceleration (5v60 sessions).
Signal is log(mean dollar-volume over 5 sessions / mean dollar-volume over 60 sessions).
It tests whether an asset attracting an unusual recent participation surge earns a
subsequent cross-sectional premium. Uses only data through 2032-01-07.
"""
import os, glob, json, re
import numpy as np, pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-01-07')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d[['close','volume']].astype(float)
dat={a:load(a) for a in A}; C=pd.DataFrame({a:dat[a].close for a in A}).loc[:END]; V=pd.DataFrame({a:dat[a].volume for a in A}).reindex(C.index)
DV=(C*V).replace([np.inf,-np.inf],np.nan)
F=np.log(DV.rolling(5,min_periods=4).mean()/DV.rolling(60,min_periods=40).mean()).replace([np.inf,-np.inf],np.nan)
def evaluate(h):
 y=C.shift(-h).div(C)-1; vals=[]; counts=[]
 for dt in F.index[:-h]:
  z=pd.concat([F.loc[dt].rename('x'),y.loc[dt].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): vals.append((dt,q));counts.append(len(z))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(counts))}
S={};M={}
for h in [1,5,10,20]:
 S[h],M[h]=evaluate(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
for lab,mask in [('2020_2021',S[10].index.year<=2021),('2022_2023',S[10].index.year.isin([2022,2023])),('2024_2026',S[10].index.year.isin([2024,2025,2026])),('2027_2030',S[10].index.year.isin([2027,2028,2029,2030])),('2031_ytd',S[10].index.year==2031),('recent_6m',S[10].index>=END-pd.Timedelta(days=183))]:
 s=S[10][mask]; print('REGIME10',lab,'dates',len(s),'ic',float(s.mean()),'icir',float(s.mean()/s.std(ddof=1)),'hit',float((s>0).mean()))
# rank stability
r=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):r.append(q)
# Resolve artifact by normalized significant tokens, including legacy naming changes.
def norm(x): return re.sub('[^a-z0-9]','',x.lower().replace('miner1','').replace('miner2','').replace('miner3',''))
def artifact(fid):
 key=norm(fid); paths=glob.glob('scripts/*_signal.pkl'); exact=[p for p in paths if key in norm(os.path.basename(p))]
 if exact:return max(exact,key=os.path.getmtime)
 # known current alias: persisted state_gated volatility is stored as state_gated_inverse volatility
 if 'stategatedvolatilityexpansion' in key:
  q=[p for p in paths if 'stategatedinversevolatilityexpansion' in norm(os.path.basename(p))]
  if q:return max(q,key=os.path.getmtime)
 return None
complete=True;mx=0.;peer=None;cells=0; audited=0; evidence={}
for p in glob.glob('factors/*.json'):
 d=json.load(open(p)); fid=d.get('factor_id','')
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 audited+=1; ap=artifact(fid)
 if not ap:
  complete=False;evidence[fid]={'rho':None,'cells':0};print('LIB',fid,'MISSING');continue
 try:
  L=pd.read_pickle(ap).reindex(index=F.index,columns=A); z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna();q=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 rho=float(q) if np.isfinite(q) else None; evidence[fid]={'rho':rho,'cells':len(z),'artifact':ap}; complete &= rho is not None
 if rho is not None and abs(rho)>mx:mx=abs(rho);peer=fid;cells=len(z)
 print('LIB',fid,'cells',len(z),'rho',rho)
out={'period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'signal_coverage':float(F.notna().mean().mean()),'mean_active_names_per_date':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(r)),'turnover_proxy':float(1-np.mean(r)),'metrics':M,'max_abs_library_correlation':mx if complete else None,'observed_max_abs_library_correlation':mx,'most_correlated':peer,'common_cells_most':cells,'library_evidence_complete':complete,'library_factors_compared':audited,'library_evidence':evidence}
print('SUMMARY',json.dumps(out,sort_keys=True));F.to_pickle('scripts/miner_2_20320108_liquidity_acceleration_5v60obs_signal.pkl')
