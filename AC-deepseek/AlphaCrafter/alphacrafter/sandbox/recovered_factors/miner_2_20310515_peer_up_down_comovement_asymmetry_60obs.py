"""miner_2: peer co-movement asymmetry: relative correlation to peers in up vs down markets."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2031-05-14')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
C=pd.DataFrame({a:load(a) for a in A}).sort_index(); R=np.log(C).diff(); market=R.median(axis=1)
# In each trailing 60-session window, calculate an asset's correlation with its peer basket separately on up and down cross-asset days.
# High values identify assets that integrate in risk-on markets but decouple when the broad cross-section falls.
peer=pd.DataFrame({a:R.drop(columns=a).mean(axis=1) for a in A})
def asym(a):
 out=[]
 for i in range(len(R)):
  w=R.index[max(0,i-59):i+1]; up=market.loc[w]>0; dn=market.loc[w]<0
  x=R.loc[w,a]; p=peer.loc[w,a]
  cu=x[up].corr(p[up]) if up.sum()>=12 else np.nan
  cd=x[dn].corr(p[dn]) if dn.sum()>=12 else np.nan
  out.append(cu-cd if np.isfinite(cu) and np.isfinite(cd) else np.nan)
 return pd.Series(out,index=R.index)
F=pd.DataFrame({a:asym(a) for a in A}).loc[:END]
def metrics(X,h):
 y=(C.shift(-h)/C-1).reindex(X.index); vals=[];ns=[]
 for d in X.index:
  z=pd.concat([X.loc[d].rename('x'),y.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.x,z.y).statistic
   if np.isfinite(q):vals.append((d,float(q)));ns.append(len(z))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/sd),'ic_hit_ratio':float((s>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(s))),'ic_dates':len(s),'mean_valid_instruments':float(np.mean(ns))}
for direction,X in [('asymmetry',F),('inverse_asymmetry',-F)]:
 for h in (1,5,10,20):
  s,m=metrics(X,h);print('HORIZON',direction,h,json.dumps(m,sort_keys=True))
  if h==5:
   for lab,mask in [('2020_2021',s.index.year<=2021),('2022_2023',s.index.year.isin([2022,2023])),('2024_2026',s.index.year.isin([2024,2025,2026])),('2027_2030',s.index.year.isin([2027,2028,2029,2030])),('2031_YTD',s.index.year==2031)]:
    q=s[mask];print('REGIME',direction,lab,len(q),float(q.mean()) if len(q) else None,float(q.mean()/q.std(ddof=1)) if len(q)>1 else None,float((q>0).mean()) if len(q) else None)
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except Exception:pass
files=glob.glob('scripts/*_signal.pkl'); ev={};mx=0;who=None;complete=True
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_',''); matches=[p for p in files if key in os.path.basename(p)]
 if not matches: complete=False;ev[fid]={'rho':None,'common_signal_cells':0};continue
 p=max(matches,key=os.path.getmtime)
 try:
  L=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('x'),L.stack().rename('l')],axis=1).dropna(); q=float(spearmanr(z.x,z.l).statistic) if len(z)>=8 else np.nan
 except Exception: z=pd.DataFrame();q=np.nan
 ev[fid]={'rho':q if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
 print('LIBRARY_CORR',fid,len(z),q)
print('SUMMARY',json.dumps({'factor':'peer_up_down_comovement_asymmetry_60obs','period':[str(F.index.min().date()),str(END.date())],'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'implied_rank_turnover':float(1-np.mean(st)),'effective_library':len(active),'correlation_evidence_complete':complete,'max_abs_library_correlation':mx if complete else None,'observed_max':mx,'most_correlated':who,'evidence':ev},sort_keys=True))
F.to_pickle('scripts/miner_2_20310515_peer_up_down_comovement_asymmetry_60obs_signal.pkl')
