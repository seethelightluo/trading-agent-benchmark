"""miner_2 20280323: upside-participation acceleration, one interpretable candidate."""
import os,json,glob
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-03-22')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({a:load(a) for a in A}).sort_index().loc[:END]; r=P.pct_change()
# Recent fraction of advancing sessions less its established 40-session participation rate.
# Separates breadth of participation from return magnitude, trend, and tail concentration.
F=(r.gt(0).rolling(10,min_periods=8).mean()-r.gt(0).rolling(40,min_periods=30).mean())
def met(h):
 R=P.shift(-h)/P-1; out=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append((d,float(q)));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments_per_ic_date':float(np.mean(ns))}
M={}
for h in [1,5,10,20]:
 x,M[h]=met(h); print('HORIZON',h,json.dumps(M[h]))
x,_=met(5)
for lab,mask in [('2020_2021',x.index.year<=2021),('2022_2023',x.index.year.isin([2022,2023])),('2024_2025',x.index.year.isin([2024,2025])),('2026_2028',x.index.year>=2026)]:
 y=x[mask]; print('REGIME_5D',lab,len(y),float(y.mean()),float(y.mean()/y.std(ddof=1)),float((y>0).mean()))
paths={}; admitted=[]
for jf in glob.glob('factors/*.json'):
 if jf.endswith('.bak'):continue
 d=json.load(open(jf)); fid=d.get('factor_id'); admitted.append(fid)
 c=glob.glob('scripts/*'+fid+'*_signal.pkl')+glob.glob('scripts/*'+fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')+'*_signal.pkl')
 if c: paths[fid]=max(c,key=os.path.getmtime)
 else: print('MISSING',fid)
mx=0.;who=None; evidence={}; complete=len(paths)==len(admitted)
for fid,pth in paths.items():
 try:
  L=pd.read_pickle(pth);L.index=pd.to_datetime(L.index)
  z=pd.concat([F.stack(),L.reindex(index=F.index,columns=A).stack()],axis=1).dropna()
  q=float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic) if len(z)>=8 else None
 except Exception as e: q=None;print('ERR',fid,repr(e))
 evidence[fid]={'rho':q,'common_cells':len(z) if 'z' in locals() else 0};print('LIBRARY_CORR',fid,evidence[fid])
 if q is None:complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('SUMMARY',json.dumps({'period':f'{F.index.min().date()} to {END.date()}','panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'max_abs_library_correlation':mx if complete else None,'observed_max_abs_library_correlation':mx,'most_correlated':who,'complete_library_evidence':complete,'admitted_factors':len(admitted),'artifacts_found':len(paths),'correlation_evidence':evidence,'decay':M},default=str))
F.to_pickle('scripts/miner_2_20280323_upside_participation_acceleration_10v40obs_signal.pkl')
