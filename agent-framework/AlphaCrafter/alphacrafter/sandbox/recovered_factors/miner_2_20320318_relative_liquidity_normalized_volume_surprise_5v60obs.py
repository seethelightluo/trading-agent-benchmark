"""One-factor validation: relative liquidity-normalized volume surprise (5 vs 60 sessions).
Signal is log(mean volume over 5 sessions / mean volume over 60 sessions), cross-sectionally
centered each date. It tests whether asset-specific participation acceleration predicts
near-term relative returns, independently of price-path signals. Cutoff: prior completed
session for the 2032-03-18 decision.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-03-17')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d[['close','volume']].astype(float)
p={a:load(a) for a in A}
C=pd.DataFrame({a:p[a]['close'] for a in A}).sort_index().loc[:END]
V=pd.DataFrame({a:p[a]['volume'] for a in A}).sort_index().reindex(C.index)
# Both rolling means require observed positive volume; centering removes persistent asset scale.
v5=V.where(V>0).rolling(5,min_periods=4).mean(); v60=V.where(V>0).rolling(60,min_periods=45).mean()
F=np.log(v5/v60).replace([np.inf,-np.inf],np.nan)
F=F.sub(F.mean(axis=1),axis=0) # relative participation rather than absolute liquidity

def evaluate(h):
 y=C.shift(-h).div(C)-1; out=[]; nn=[]
 for d in F.index[:-h]:
  z=pd.concat([F.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q): out.append((d,q)); nn.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(nn))}
S={};M={}
for h in [1,5,10,20]:
 S[h],M[h]=evaluate(h); print('HORIZON',h,json.dumps(M[h],sort_keys=True))
for name,mask in [('2024_2026',S[1].index.year.isin([2024,2025,2026])),('2027_2030',S[1].index.year.isin([2027,2028,2029,2030])),('2031_2032',S[1].index.year.isin([2031,2032])),('recent_6m',S[1].index>=END-pd.Timedelta(days=183))]:
 s=S[1][mask]; print('REGIME1',name,'dates',len(s),'ic',float(s.mean()),'icir',float(s.mean()/s.std(ddof=1)),'hit',float((s>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):st.append(q)
complete=True; mx=0.;peer=None;pcells=0;audited=0;evidence={}
for path in glob.glob('factors/*.json'):
 d=json.load(open(path)); fid=d.get('factor_id','')
 if d.get('validation',{}).get('status')!='EFFECTIVE': continue
 audited+=1; key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 paths=glob.glob('scripts/*'+key+'*_signal.pkl')
 if not paths:
  complete=False;evidence[fid]={'rho':None,'cells':0};print('LIB',fid,'MISSING');continue
 try:
  L=pd.read_pickle(max(paths,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna();q=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 rho=float(q) if np.isfinite(q) else None;evidence[fid]={'rho':rho,'cells':len(z)}
 if rho is None: complete=False
 elif abs(rho)>mx: mx=abs(rho);peer=fid;pcells=len(z)
 print('LIB',fid,'cells',len(z),'rho',rho)
summary={'period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'signal_coverage':float(F.notna().mean().mean()),'mean_active_names_per_date':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'turnover_proxy':float(1-np.mean(st)),'metrics':M,'max_abs_library_correlation':mx if complete else None,'observed_max_abs_library_correlation':mx,'most_correlated':peer,'common_cells_most':pcells,'library_evidence_complete':complete,'library_factors_compared':audited,'library_evidence':evidence}
print('SUMMARY',json.dumps(summary,sort_keys=True)); F.to_pickle('scripts/miner_2_20320318_relative_liquidity_normalized_volume_surprise_5v60obs_signal.pkl')
