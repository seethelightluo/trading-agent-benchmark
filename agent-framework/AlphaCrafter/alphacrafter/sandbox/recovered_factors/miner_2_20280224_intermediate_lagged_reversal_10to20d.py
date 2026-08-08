"""miner_2 research: intermediate (10-20d lag) cross-asset reversal, 2028-02-24."""
import os,json,glob
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2028-02-23')
def load(a):
 return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
P=pd.DataFrame({a:load(a) for a in A}).sort_index().loc[:END]
# Pre-specified reversal: negative return during days t-20 through t-10, deliberately excluding the most recent 10 days.
# This isolates intermediate-horizon overreaction from short-term reversal and current trend factors.
F=-(P.shift(10).div(P.shift(20)).sub(1))
def getic(h):
 future=P.shift(-h).div(P).sub(1); out=[]; n=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),future.loc[d].rename('r')],axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   out.append((d,float(spearmanr(z.f,z.r).statistic)));n.append(len(z))
 x=pd.Series(dict(out)); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments':float(np.mean(n))}
M={}
for h in [1,5,10,20]:
 x,M[h]=getic(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
 for lab,mask in [('2020_2021',x.index.year<=2021),('2022_2023',x.index.year.isin([2022,2023])),('2024_2025',x.index.year.isin([2024,2025])),('2026_2028',x.index.year>=2026)]:
  y=x[mask];print('REGIME',h,lab,'dates',len(y),'IC',float(y.mean()),'ICIR',float(y.mean()/y.std(ddof=1)),'hit',float((y>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p))
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except:pass
found={}; missing=[]
for fid in active:
 q=glob.glob('scripts/*'+fid+'*_signal.pkl')
 if q:found[fid]=max(q,key=os.path.getmtime)
 else:missing.append(fid)
cor={};mx=-1;who=None
for fid,p in found.items():
 L=pd.read_pickle(p);L.index=pd.to_datetime(L.index)
 z=pd.concat([F.stack().rename('f'),L.reindex(index=F.index,columns=A).stack().rename('l')],axis=1).dropna()
 rho=float(spearmanr(z.f,z.l).statistic) if len(z)>=8 else None
 cor[fid]={'rho':rho,'common_signal_cells':len(z)}
 if rho is not None and abs(rho)>mx:mx=abs(rho);who=fid
 print('LIBRARY_CORR',fid,len(z),rho)
print('FACTOR intermediate_lagged_reversal_10to20d')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(axis=1).mean()),'rank_stability_1d',float(np.mean(st)))
print('DECAY',json.dumps(M,sort_keys=True))
print('LIBRARY_AUDIT','active',len(active),'artifacts_found',len(found),'missing',missing,'max_abs_library_correlation',mx if not missing else None,'most_correlated',who,'evidence',json.dumps(cor,sort_keys=True))
F.to_pickle('scripts/miner_2_20280224_intermediate_lagged_reversal_10to20d_signal.pkl')
