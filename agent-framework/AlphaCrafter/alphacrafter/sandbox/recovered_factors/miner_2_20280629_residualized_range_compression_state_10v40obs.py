"""miner_2 20280629: residualized range-compression state, one candidate.
Range-compression multiplier is cross-sectionally residualized against the 10-to-20d
intermediate return, isolating a volatility-state signal rather than trend exposure.
"""
import os,json,glob
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-06-28')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index() for a in A}
P=pd.DataFrame({a:D[a].close.astype(float) for a in A}).sort_index().loc[:END]
H=pd.DataFrame({a:D[a].high.astype(float) for a in A}).sort_index().reindex(P.index); L=pd.DataFrame({a:D[a].low.astype(float) for a in A}).sort_index().reindex(P.index)
rng=(H-L)/P
trend=P.shift(10)/P.shift(20)-1
compression=1-rng.rolling(10,min_periods=10).mean()/rng.rolling(40,min_periods=40).mean()
# Per-date OLS residual: compression - (a + b*intermediate trend), with >=8 names.
F=pd.DataFrame(np.nan,index=P.index,columns=A)
for d in P.index:
 z=pd.concat([compression.loc[d].rename('c'),trend.loc[d].rename('t')],axis=1).dropna()
 if len(z)>=8 and z.t.nunique()>1:
  b,a=np.polyfit(z.t,z.c,1); F.loc[d,z.index]=z.c-(a+b*z.t)
def met(h):
 R=P.shift(-h)/P-1; vals=[]; nn=[]
 for d in F.index:
  z=pd.concat([F.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): vals.append((d,float(q)));nn.append(len(z))
 x=pd.Series(dict(vals),dtype=float); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments_per_ic_date':float(np.mean(nn))}
M={}
for h in [1,5,10,20]:
 x,M[h]=met(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
x,_=met(5)
for lab,mask in [('2020_2021',x.index.year<=2021),('2022_2023',x.index.year.isin([2022,2023])),('2024_2025',x.index.year.isin([2024,2025])),('2026_2028',x.index.year>=2026),('recent_120',x.index.isin(x.index[-120:]))]:
 y=x[mask];print('REGIME_5D',lab,'dates',len(y),'IC',float(y.mean()) if len(y) else None,'ICIR',float(y.mean()/y.std(ddof=1)) if len(y)>1 else None,'hit',float((y>0).mean()) if len(y) else None)
paths={}; admitted=[]
for jf in glob.glob('factors/*.json'):
 try: rec=json.load(open(jf))
 except: continue
 if rec.get('validation',{}).get('status')=='EFFECTIVE':
  fid=rec['factor_id']; admitted.append(fid)
  choices=glob.glob('scripts/*'+fid+'*_signal.pkl')+glob.glob('scripts/*'+fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')+'*_signal.pkl')
  if choices: paths[fid]=max(choices,key=os.path.getmtime)
  else: print('MISSING',fid)
complete=len(paths)==len(admitted); evidence={};mx=0.;who=None
for fid,p in paths.items():
 try:
  Q=pd.read_pickle(p);Q.index=pd.to_datetime(Q.index)
  z=pd.concat([F.stack().rename('f'),Q.reindex(index=F.index,columns=A).stack().rename('q')],axis=1).dropna()
  rho=float(spearmanr(z.f,z.q).statistic) if len(z)>=8 and z.f.nunique()>1 and z.q.nunique()>1 else None
 except Exception as e: rho=None;z=pd.DataFrame();print('ERR',fid,repr(e))
 evidence[fid]={'rho':rho,'common_cells':len(z)};print('LIBRARY_CORR',fid,evidence[fid])
 if rho is None:complete=False
 elif abs(rho)>mx:mx=abs(rho);who=fid
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
print('SUMMARY',json.dumps({'factor_id':'residualized_range_compression_state_10v40obs','period':str(F.index.min().date())+' to '+str(END.date()),'panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability_1d':float(np.mean(st)),'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'complete_library_evidence':complete,'admitted_factors':len(admitted),'artifacts_found':len(paths),'decay':M,'evidence':evidence},default=str))
F.to_pickle('scripts/miner_2_20280629_residualized_range_compression_state_10v40obs_signal.pkl')
