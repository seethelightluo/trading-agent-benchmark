"""miner_2 20281005: volatility-neutral intraday pressure reversal, one candidate."""
import os,json,glob
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-10-04')
D={a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index() for a in A}
P=pd.DataFrame({a:D[a].close.astype(float) for a in A}).sort_index().loc[:END]
O=pd.DataFrame({a:D[a].open.astype(float) for a in A}).reindex(P.index)
# Raw signal rewards sustained negative open-to-close pressure (a rebound candidate).
# Each date it is cross-sectionally residualized against trailing realized volatility,
# explicitly removing the volatility component that contaminated prior close-location tests.
raw=-(P/O-1).rolling(5,min_periods=4).mean()
vol=P.pct_change().rolling(20,min_periods=12).std()
F=pd.DataFrame(index=P.index,columns=A,dtype=float)
for d in P.index:
 z=pd.concat([raw.loc[d],vol.loc[d]],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,1].std()>0:
  x=z.iloc[:,1].values; y=z.iloc[:,0].values
  beta=np.cov(x,y,ddof=1)[0,1]/np.var(x,ddof=1)
  F.loc[d,z.index]=y-(y.mean()-beta*x.mean())-beta*x

def met(h):
 R=P.shift(-h)/P-1;out=[];ns=[]
 for d in F.index:
  z=pd.concat([F.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append((d,float(q)));ns.append(len(z))
 x=pd.Series(dict(out),dtype=float);sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments_per_ic_date':float(np.mean(ns))}
M={}
for h in [1,5,10,20]:
 x,M[h]=met(h);print('HORIZON',h,json.dumps(M[h]))
x,_=met(5)
for lab,mask in [('2020_2021',x.index.year<=2021),('2022_2023',x.index.year.isin([2022,2023])),('2024_2025',x.index.year.isin([2024,2025])),('2026_2028',x.index.year>=2026)]:
 y=x[mask];print('REGIME_5D',lab,len(y),float(y.mean()) if len(y) else None,float(y.mean()/y.std(ddof=1)) if len(y)>1 else None,float((y>0).mean()) if len(y) else None)
paths={};admitted=[]
for jf in glob.glob('factors/*.json'):
 d=json.load(open(jf));fid=d.get('factor_id')
 if d.get('validation',{}).get('status')=='EFFECTIVE':
  admitted.append(fid); c=glob.glob('scripts/*'+fid+'*_signal.pkl')+glob.glob('scripts/*'+fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','')+'*_signal.pkl')
  if c:paths[fid]=max(c,key=os.path.getmtime)
  else:print('MISSING',fid)
mx=0.;who=None;ev={};complete=len(paths)==len(admitted)
for fid,pth in paths.items():
 try:
  Z=pd.read_pickle(pth);Z.index=pd.to_datetime(Z.index);z=pd.concat([F.stack(),Z.reindex(index=F.index,columns=A).stack()],axis=1).dropna();q=float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic) if len(z)>=8 else None
 except Exception as e:q=None;z=pd.DataFrame();print('ERR',fid,repr(e))
 ev[fid]={'rho':q,'common_cells':len(z)};print('LIBRARY_CORR',fid,ev[fid])
 if q is None:complete=False
 elif abs(q)>mx:mx=abs(q);who=fid
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('SUMMARY',json.dumps({'factor':'volatility_neutral_intraday_pressure_reversal_5v20obs','period':f'{F.index.min().date()} to {END.date()}','panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'complete_library_evidence':complete,'admitted_factors':len(admitted),'artifacts_found':len(paths),'correlation_evidence':ev,'decay':M},default=str))
F.to_pickle('scripts/miner_2_20281005_volatility_neutral_intraday_pressure_reversal_5v20obs_signal.pkl')
