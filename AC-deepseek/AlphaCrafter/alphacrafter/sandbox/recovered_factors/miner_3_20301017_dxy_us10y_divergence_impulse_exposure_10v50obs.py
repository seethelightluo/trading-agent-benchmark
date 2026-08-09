"""miner_3: dollar/rate divergence impulse exposure; one interpretable macro-transmission factor."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2030-10-16')
def close(p): return pd.read_csv(p,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:close('../persistent/stock_data/'+a+'.csv') for a in A}).sort_index()
dxy=close('../persistent/index_data/DXY.csv').reindex(C.index).ffill()
usy=close('../persistent/stock_data/US10Y.csv').reindex(C.index).ffill()
r=np.log(C).diff(); dr=np.log(dxy).diff(); yr=np.log(usy).diff()
# A positive shock means dollar strength unusually exceeds the simultaneous rate move.
dz=(dr-dr.rolling(50,min_periods=35).mean())/dr.rolling(50,min_periods=35).std()
yz=(yr-yr.rolling(50,min_periods=35).mean())/yr.rolling(50,min_periods=35).std()
div=dz-yz
beta=r.rolling(50,min_periods=35).cov(div).div(div.rolling(50,min_periods=35).var(),axis=0)
F=beta.mul(-div.rolling(10,min_periods=10).sum(),axis=0).loc[:END]
def calc(h):
 future=(C.shift(-h)/C-1).reindex(F.index);out=[];width=[]
 for d in F.index:
  z=pd.concat([F.loc[d].rename('f'),future.loc[d].rename('y')],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.f,z.y).statistic
   if np.isfinite(q):out.append((d,float(q)));width.append(len(z))
 ic=pd.Series(dict(out)); sd=ic.std(ddof=1)
 return ic,{'daily_paper_ic':float(ic.mean()),'daily_paper_icir':float(ic.mean()/sd),'ic_hit_ratio':float((ic>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(ic))),'ic_dates':len(ic),'mean_valid_instruments':float(np.mean(width))}
M={}
for h in (1,5,10,20): _,M[h]=calc(h);print('HORIZON',h,json.dumps(M[h],sort_keys=True))
ic,_=calc(10)
for lab,mask in [('2020_2021',ic.index.year<=2021),('2022_2023',ic.index.year.isin([2022,2023])),('2024_2026',ic.index.year.isin([2024,2025,2026])),('2027_2030',ic.index.year>=2027)]:
 x=ic[mask];print('REGIME_10D',lab,'dates',len(x),'IC',float(x.mean()),'ICIR',float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,'hit',float((x>0).mean()))
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8: st.append(float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic))
active=[json.load(open(p))['factor_id'] for p in glob.glob('factors/*.json') if '_deprecated' not in p]
mx=0.;most=None;evidence={}
for fid in active:
 key=fid.replace('miner_1_','').replace('miner_2_','').replace('miner_3_','');files=[p for p in glob.glob('scripts/*_signal.pkl') if key in os.path.basename(p)]
 if not files: evidence[fid]={'rho':None,'common_signal_cells':0};mx=np.inf;continue
 p=max(files,key=os.path.getmtime)
 try:
  lib=pd.read_pickle(p).reindex(index=F.index,columns=A);z=pd.concat([F.stack().rename('a'),lib.stack().rename('b')],axis=1).dropna();q=float(spearmanr(z.a,z.b).statistic) if len(z)>=8 else np.nan
 except Exception:z=pd.DataFrame();q=np.nan
 evidence[fid]={'rho':q if np.isfinite(q) else None,'common_signal_cells':len(z),'file':p}
 if not np.isfinite(q):mx=np.inf
 elif abs(q)>mx:mx=abs(q);most=fid
 print('LIBRARY_CORR',fid,'cells',len(z),'spearman',q)
print('FACTOR dxy_us10y_divergence_impulse_exposure_10v50obs')
print('PERIOD',F.index.min().date(),END.date(),'panel_dates',len(F),'coverage',float(F.notna().mean().mean()),'mean_names',float(F.notna().sum(1).mean()),'rank_stability',float(np.mean(st)),'implied_turnover',float(1-np.mean(st)))
print('MAX_ABS_LIBRARY_CORRELATION',mx,'MOST',most,'EVIDENCE',json.dumps(evidence,sort_keys=True));F.to_pickle('scripts/miner_3_20301017_dxy_us10y_divergence_impulse_exposure_10v50obs_signal.pkl')
