"""One-factor validation: liquidity-shock-weighted short-term reversal.
At each date the signal is minus the five-session return multiplied by the asset's
20-session standardized log-volume surprise. This tests whether recent moves
accompanied by unusually high participation are more likely to reverse.
Cutoff is the prior completed session to the 2032-02-05 decision date.
"""
import os, glob, json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2032-02-04')
def load(a):
 d=pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d[['close','volume']].astype(float)
p={a:load(a) for a in A}
C=pd.DataFrame({a:p[a]['close'] for a in A}).sort_index().loc[:END]
V=pd.DataFrame({a:p[a]['volume'] for a in A}).sort_index().reindex(C.index)
lv=np.log(V.where(V>0)); vz=(lv-lv.rolling(20,min_periods=15).mean()).div(lv.rolling(20,min_periods=15).std(ddof=1))
F=(-C.pct_change(5)*vz).replace([np.inf,-np.inf],np.nan)
def evaluate(h):
 y=C.shift(-h).div(C)-1; out=[]; nn=[]
 for d in F.index[:-h]:
  z=pd.concat([F.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q):out.append((d,q));nn.append(len(z))
 s=pd.Series(dict(out)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(nn))}
S={};M={}
for h in [1,5,10,20]:
 S[h],M[h]=evaluate(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
for name,mask in [('2024_2026',S[5].index.year.isin([2024,2025,2026])),('2027_2030',S[5].index.year.isin([2027,2028,2029,2030])),('2031',S[5].index.year==2031),('recent_6m',S[5].index>=END-pd.Timedelta(days=183))]:
 s=S[5][mask]; print('REGIME5',name,'dates',len(s),'ic',float(s.mean()),'icir',float(s.mean()/s.std(ddof=1)),'hit',float((s>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:
  q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
  if np.isfinite(q):st.append(q)
complete=True; mx=0.;peer=None;pcells=0;audited=0;evidence={}
for path in glob.glob('factors/*.json'):
 d=json.load(open(path)); fid=d.get('factor_id','')
 if d.get('validation',{}).get('status')!='EFFECTIVE':continue
 audited+=1; key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')
 paths=glob.glob('scripts/*'+key+'*_signal.pkl')
 if not paths:
  complete=False;evidence[fid]={'rho':None,'cells':0};print('LIB',fid,'MISSING');continue
 try:
  L=pd.read_pickle(max(paths,key=os.path.getmtime)).reindex(index=F.index,columns=A)
  z=pd.concat([F.stack().rename('x'),L.stack().rename('y')],axis=1).dropna();q=spearmanr(z.x,z.y).statistic if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 rho=float(q) if np.isfinite(q) else None;evidence[fid]={'rho':rho,'cells':len(z)}
 if rho is None:complete=False
 elif abs(rho)>mx:mx=abs(rho);peer=fid;pcells=len(z)
 print('LIB',fid,'cells',len(z),'rho',rho)
summary={'period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'signal_coverage':float(F.notna().mean().mean()),'mean_active_names_per_date':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'turnover_proxy':float(1-np.mean(st)),'metrics':M,'max_abs_library_correlation':mx if complete else None,'observed_max_abs_library_correlation':mx,'most_correlated':peer,'common_cells_most':pcells,'library_evidence_complete':complete,'library_factors_compared':audited,'library_evidence':evidence}
print('SUMMARY',json.dumps(summary,sort_keys=True));F.to_pickle('scripts/miner_2_20320205_liquidity_shock_weighted_reversal_5v20obs_signal.pkl')
