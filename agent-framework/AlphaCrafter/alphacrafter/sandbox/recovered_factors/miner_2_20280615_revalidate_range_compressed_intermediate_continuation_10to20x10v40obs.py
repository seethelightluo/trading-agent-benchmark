"""miner_2 20280615: quarterly revalidation of one admitted factor, range-compressed intermediate continuation."""
import os,json,glob
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-06-14')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index() for a in A}
P=pd.DataFrame({a:D[a].close.astype(float) for a in A}).sort_index().loc[:END]
H=pd.DataFrame({a:D[a].high.astype(float) for a in A}).sort_index().reindex(P.index)
L=pd.DataFrame({a:D[a].low.astype(float) for a in A}).sort_index().reindex(P.index)
# Exact persisted definition. Values are retained only in the common live-era panel so no unavailable-market proxy is introduced.
rng=(H-L)/P
F=(P.shift(10)/P.shift(20)-1)*(1-rng.rolling(10,min_periods=10).mean()/rng.rolling(40,min_periods=40).mean())
def metrics(h):
 R=P.shift(-h)/P-1; vals=[]; nums=[]
 for d in F.index:
  z=pd.concat([F.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append((d,float(q)));nums.append(len(z))
 x=pd.Series(dict(vals),dtype=float); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments_per_ic_date':float(np.mean(nums))}
M={}
for h in [1,5,10,20]:
 x,M[h]=metrics(h); print('HORIZON',h,json.dumps(M[h],sort_keys=True))
x,_=metrics(5)
for label,mask in [('2026',x.index.year==2026),('2027',x.index.year==2027),('2028_ytd',x.index.year==2028),('last_120_ic_dates',x.index.isin(x.index[-120:]))]:
 y=x[mask]; print('REGIME_5D',label,'dates',len(y),'IC',float(y.mean()),'ICIR',float(y.mean()/y.std(ddof=1)) if len(y)>1 else None,'hit',float((y>0).mean()))
# Spearman library evidence uses all common valid cells for each admitted signal.
paths={}; admitted=[]
for jf in glob.glob('factors/*.json'):
 try: rec=json.load(open(jf))
 except: continue
 if rec.get('validation',{}).get('status')=='EFFECTIVE':
  fid=rec['factor_id'];admitted.append(fid)
  choices=glob.glob('scripts/*'+fid+'*_signal.pkl')+glob.glob('scripts/*'+fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')+'*_signal.pkl')
  if choices: paths[fid]=max(choices,key=os.path.getmtime)
  else: print('MISSING',fid)
complete=len(paths)==len(admitted); evidence={};mx=0.;who=None
for fid,pth in paths.items():
 try:
  Q=pd.read_pickle(pth);Q.index=pd.to_datetime(Q.index)
  z=pd.concat([F.stack().rename('factor'),Q.reindex(index=F.index,columns=A).stack().rename('library')],axis=1).dropna()
  rho=float(spearmanr(z.factor,z.library).statistic) if len(z)>=8 and z.factor.nunique()>1 and z.library.nunique()>1 else None
 except Exception as e: rho=None;z=pd.DataFrame();print('ERR',fid,repr(e))
 evidence[fid]={'rho':rho,'common_cells':len(z)};print('LIBRARY_CORR',fid,evidence[fid])
 if rho is None: complete=False
 elif abs(rho)>mx:mx=abs(rho);who=fid
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('SUMMARY',json.dumps({'period':str(F.index.min().date())+' to '+str(END.date()),'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'complete_library_evidence':complete,'admitted_factors':len(admitted),'artifacts_found':len(paths),'decay':M,'evidence':evidence},default=str))
F.to_pickle('scripts/miner_2_20280615_revalidate_range_compressed_intermediate_continuation_10to20x10v40obs_signal.pkl')
