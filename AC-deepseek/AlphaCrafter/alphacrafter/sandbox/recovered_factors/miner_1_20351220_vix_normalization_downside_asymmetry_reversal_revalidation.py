"""Miner 1 revalidation and complete-library correlation audit, 2035-12-19."""
import os,glob,json
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-12-19')
def load(path):
 return pd.read_csv(path,parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()['close'].astype(float)
C=pd.DataFrame({a:load('../persistent/stock_data/'+a+'.csv') for a in A}).loc[:END]; R=C.pct_change()
V=load('../persistent/index_data/VIX.csv').reindex(C.index).ffill()
down=R.clip(upper=0).pow(2).rolling(20,min_periods=20).mean(); up=R.clip(lower=0).pow(2).rolling(20,min_periods=20).mean()
state=(V<V.rolling(20,min_periods=15).mean())&(V.rolling(60,min_periods=45).mean()>V.rolling(252,min_periods=180).median())
F=(down/(down+up).replace(0,np.nan)).where(state,0.0)
F.to_pickle('scripts/miner_1_20351220_vix_normalization_downside_asymmetry_reversal_20x20x60obs_signal.pkl')
def metrics(h):
 y=C.shift(-h).div(C)-1; vals=[]; nn=[]
 for d in F.index:
  z=pd.concat((F.loc[d].rename('f'),y.loc[d].rename('r')),axis=1).dropna()
  if len(z)>=8 and z.f.nunique()>1:
   q=spearmanr(z.f,z.r).statistic
   if np.isfinite(q): vals.append((d,q));nn.append(len(z))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,dict(daily_paper_ic=float(s.mean()),daily_paper_icir=float(s.mean()/sd),ic_hit_ratio=float((s>0).mean()),ic_standard_error=float(sd/np.sqrt(len(s))),ic_dates=len(s),mean_valid_instruments=float(np.mean(nn)))
M={}
for h in (1,5,10,20,40):
 s,M[h]=metrics(h);print('H',h,json.dumps(M[h],sort_keys=True))
s,_=metrics(10)
for n,lo,hi in [('2024_26','2024-01-01','2026-12-31'),('2027_30','2027-01-01','2030-12-31'),('2031_33','2031-01-01','2033-12-31'),('2034_recent','2034-01-01',str(END.date()))]:
 x=s.loc[lo:hi];print('REGIME',n,len(x),float(x.mean()) if len(x) else None,float(x.mean()/x.std(ddof=1)) if len(x)>1 else None,float((x>0).mean()) if len(x) else None)
# exact persisted file-stem is the authoritative artifact mapping; two legacy exceptions retain prior dated artifact.
legacy={'miner_1_20311211_state_gated_volatility_expansion_10v60obs':'scripts/miner_1_20320304_state_gated_volatility_expansion_10v60obs_signal.pkl','miner_2_20280127_standardized_jump_asymmetry_20v40obs':'scripts/miner_2_20280113_standardized_jump_asymmetry_20v40obs_signal.pkl'}
E={}; mx=-1; who=None; complete=True; active=[]
for p in glob.glob('factors/*.json'):
 j=json.load(open(p));
 if j.get('validation',{}).get('status')!='EFFECTIVE':continue
 stem=os.path.basename(p)[:-5]; active.append(stem); artifact=legacy.get(stem,'scripts/'+stem+'_signal.pkl')
 try:
  Q=pd.read_pickle(artifact).reindex(index=F.index,columns=A)
  z=pd.concat((F.stack().rename('f'),Q.stack().rename('q')),axis=1).dropna(); rho=spearmanr(z.f,z.q).statistic if len(z)>=8 else np.nan
 except Exception as e: z=pd.DataFrame();rho=np.nan
 E[stem]={'rho':float(rho) if np.isfinite(rho) else None,'common_signal_cells':len(z),'artifact':artifact}
 if not np.isfinite(rho):complete=False
 elif abs(rho)>mx:mx=abs(rho);who=stem
print('AUDIT',len(active),'complete',complete,'max_abs_library_correlation',mx,'with',who)
print('EVIDENCE',json.dumps(E,sort_keys=True))
st=[]
for i in range(1,len(F)):
 z=pd.concat((F.iloc[i-1],F.iloc[i]),axis=1).dropna()
 if z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('PANEL',C.index.min().date(),C.index.max().date(),'coverage',float(F.notna().mean().mean()),'mean_valid_names',float(F.notna().sum(axis=1).mean()),'activation',float(state.mean()),'rank_stability',float(np.nanmean(st)),'turnover',float(1-np.nanmean(st)))
print('ALL',json.dumps(M,sort_keys=True))
