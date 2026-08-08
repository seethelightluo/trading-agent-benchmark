"""miner_2 20280309: recent-return acceleration, one interpretable candidate."""
import os,json,glob
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-03-08')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({a:load(a) for a in A}).sort_index().loc[:END]
# Difference of recent 5-session trend and its 20-session average pace. Positive means trend acceleration.
F=P.pct_change(5)-(P.pct_change(20)/4)
def met(h):
 R=P.shift(-h)/P-1; out=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q): out.append((d,float(q)));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments_per_ic_date':float(np.mean(ns))}
M={}
for h in [1,5,10,20]:
 x,M[h]=met(h); print('HORIZON',h,json.dumps(M[h]))
x,_=met(5)
for lab,mask in [('2020_2021',x.index.year<=2021),('2022_2023',x.index.year.isin([2022,2023])),('2024_2025',x.index.year.isin([2024,2025])),('2026_2028',x.index.year>=2026)]:
 y=x[mask]; print('REGIME_5D',lab,len(y),float(y.mean()),float(y.mean()/y.std(ddof=1)),float((y>0).mean()))
paths={};admitted=[]
for jf in glob.glob('factors/*.json'):
 if jf.endswith('.bak'):continue
 d=json.load(open(jf));fid=d.get('factor_id');admitted.append(fid)
 c=glob.glob('scripts/*'+fid+'*_signal.pkl')+glob.glob('scripts/*'+fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')+'*_signal.pkl')
 if c: paths[fid]=max(c,key=os.path.getmtime)
 else: print('MISSING',fid)
complete=len(paths)==len(admitted);mx=0.;who=None
for fid,pth in paths.items():
 try:
  L=pd.read_pickle(pth);L.index=pd.to_datetime(L.index)
  z=pd.concat([F.stack(),L.reindex(index=F.index,columns=A).stack()],axis=1).dropna()
  q=float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic) if len(z)>=8 else None
 except Exception as e:q=None;print('ERR',fid,repr(e))
 print('LIBRARY_CORR',fid,len(z),q)
 if q is None:complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('SUMMARY',json.dumps({'period':f'{F.index.min().date()} to {END.date()}','panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'max_abs_library_correlation':mx if complete else None,'observed_max_abs_library_correlation':mx,'most_correlated':who,'complete_library_evidence':complete,'admitted_factors':len(admitted),'artifacts_found':len(paths),'decay':M}))
F.to_pickle('scripts/miner_2_20280309_trend_acceleration_5v20obs_signal.pkl')
