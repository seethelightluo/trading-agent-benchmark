"""miner_2 20271230: rolling return skewness, one candidate."""
import os,json,glob
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; END=pd.Timestamp('2027-12-29')
def load(a): return pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index().close.astype(float)
P=pd.DataFrame({a:load(a) for a in A}).sort_index().loc[:END]; r=P.pct_change()
# Negative realized skew: unusually left-tailed recent return paths are tested as a cross-asset reversal signal.
F=-r.rolling(20,min_periods=15).skew()
def met(h):
 R=P.shift(-h)/P-1;out=[];nn=[]
 for d in F.index:
  z=pd.concat([F.loc[d],R.loc[d]],axis=1).dropna()
  if len(z)>=8:
   q=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic
   if np.isfinite(q):out.append((d,float(q)));nn.append(len(z))
 x=pd.Series(dict(out),dtype=float); sd=x.std(ddof=1)
 return x,{'daily_paper_ic':float(x.mean()),'daily_paper_icir':float(x.mean()/sd),'ic_hit_ratio':float((x>0).mean()),'ic_standard_error':float(sd/np.sqrt(len(x))),'ic_dates':len(x),'mean_valid_instruments_per_ic_date':float(np.mean(nn))}
M={}
for h in [1,5,10,20]: x,M[h]=met(h);print('HORIZON',h,json.dumps(M[h]))
x,_=met(5)
for lab,mask in [('2020',x.index.year==2020),('2021_2022',x.index.year.isin([2021,2022])),('2023_2024',x.index.year.isin([2023,2024])),('2025_2027',x.index.year>=2025)]:
 y=x[mask];print('REGIME_5D',lab,len(y),float(y.mean()) if len(y) else None,float(y.mean()/y.std(ddof=1)) if len(y)>1 else None,float((y>0).mean()) if len(y) else None)
paths={}
for fp in glob.glob('scripts/*_signal.pkl'):
 n=os.path.basename(fp).replace('_signal.pkl','')
 # Resolve by exact factor_id embedded in factor JSON below rather than filename heuristics
paths={}
for jf in glob.glob('factors/*.json'):
 if jf.endswith('.bak'):continue
 d=json.load(open(jf)); fid=d.get('factor_id');
 # supplied research convention: find a pickle whose filename begins with dated prefix and contains factor suffix
 cand=glob.glob('scripts/*'+os.path.basename(jf).replace('.json','') .split('_',2)[-1]+'*_signal.pkl')
 if not cand:
  # use manually discover based on factor ID substring
  cand=glob.glob('scripts/*'+fid.replace('miner_2_','')+'*_signal.pkl')
 if cand: paths[fid]=cand[-1]
 else: print('MISSING',fid)
mx=0;who=None;complete=len(paths)==17
for n,p in paths.items():
 try:L=pd.read_pickle(p);L.index=pd.to_datetime(L.index); z=pd.concat([F.stack(),L.reindex(index=F.index,columns=A).stack()],axis=1).dropna();q=float(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic) if len(z)>=8 else None
 except Exception as e:q=None;print('ERR',n,str(e))
 print('LIBRARY_CORR',n,len(z) if 'z' in locals() else 0,q)
 if q is None:complete=False
 elif abs(q)>mx:mx=abs(q);who=n
st=[]
for i in range(1,len(F)):
 z=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(z)>=8:st.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
print('SUMMARY',json.dumps({'period':f'{F.index.min().date()} to {END.date()}','panel_dates':len(F),'coverage':float(F.notna().mean().mean()),'mean_names':float(F.notna().sum(axis=1).mean()),'rank_stability':float(np.mean(st)),'max_abs_library_correlation':mx if complete else None,'most_correlated':who,'complete':complete,'decay':M}))
F.to_pickle('scripts/miner_2_20271230_negative_realized_skewness_20obs_signal.pkl')
