"""miner_1: inverse upside cross-asset beta, 40 observations; one candidate only."""
import os,glob,json,re
from difflib import SequenceMatcher
import numpy as np,pandas as pd
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
END=pd.Timestamp('2035-07-18')
def load(sym):
 d=pd.read_csv('../persistent/stock_data/'+sym+'.csv',parse_dates=['date']).drop_duplicates('date').set_index('date').sort_index()
 return d.close.astype(float)
P=pd.DataFrame({a:load(a) for a in A}).loc[:END]; R=np.log(P).diff()
# On broad positive-return days, estimate each asset's 40-observation upside beta.
# Lower upside beta is scored higher: defensive participation distinct from downside-beta behavior.
mkt=R.mean(axis=1); up=mkt.where(mkt>0)
def ubeta(x):
 return x.rolling(40,min_periods=25).cov(up)/up.rolling(40,min_periods=25).var()
F=pd.DataFrame({a:-ubeta(R[a]) for a in A})
# Cross-sectional percentile transformation does not change IC but provides a deployment-ready signal.
F=(F.rank(axis=1,pct=True)-.5).where(lambda x:x.count(axis=1)>=8)
def metric(X,h):
 y=np.log(P.shift(-h)/P); vals=[]; ns=[]
 for d in X.index:
  q=pd.concat([X.loc[d].rename('f'),y.loc[d].rename('r')],axis=1).dropna()
  if len(q)>=8 and q.f.nunique()>1:
   v=spearmanr(q.f,q.r).statistic
   if np.isfinite(v):vals.append((d,float(v)));ns.append(len(q))
 s=pd.Series(dict(vals)); sd=s.std(ddof=1)
 return s,{'daily_paper_ic':float(s.mean()),'daily_paper_icir':float(s.mean()/(sd+1e-12)),'ic_hit_ratio':float((s>0).mean()),'ic_dates':int(len(s)),'mean_valid_instruments':float(np.mean(ns)),'ic_standard_error':float(sd/np.sqrt(len(s)))}
print('FACTOR inverse_upside_cross_asset_beta_40obs','VALIDATION_DATE',END.date(),'PERIOD',F.index.min().date(),END.date(),'ASSETS',len(A))
allm={}
for h in [1,5,10,20,40]:
 s,m=metric(F,h);allm[h]=m;print('HORIZON',h,json.dumps(m,sort_keys=True))
s,_=metric(F,20)
for lab,l,r in [('2024_2026','2024-01-01','2026-12-31'),('2027_2030','2027-01-01','2030-12-31'),('2031_2033','2031-01-01','2033-12-31'),('2034_plus','2034-01-01',str(END.date()))]:
 z=s.loc[l:r];print('REGIME_20D',lab,'dates',len(z),'IC',float(z.mean()),'ICIR',float(z.mean()/(z.std(ddof=1)+1e-12)),'hit',float((z>0).mean()))
st=[]
for i in range(1,len(F)):
 q=pd.concat([F.iloc[i-1],F.iloc[i]],axis=1).dropna()
 if len(q)>=8: st.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic)
print('PANEL_COVERAGE',float(F.notna().mean().mean()),'SIGNAL_DATES',int(F.notna().any(axis=1).sum()),'MEAN_NAMES',float(F.count(axis=1).mean()),'RANK_STABILITY_1D',float(np.mean(st)),'TURNOVER_PROXY',float(1-np.mean(st)))
# Evidence audit: match each factor id to its best existing signal artifact; reject low-confidence/malformed matches.
active=[]
for p in glob.glob('factors/*.json'):
 try:
  j=json.load(open(p));
  if j.get('validation',{}).get('status')=='EFFECTIVE':active.append(j['factor_id'])
 except:pass
files=glob.glob('scripts/*_signal.pkl'); base=F.stack().rename('x'); ev=[]; missing=[]; mapping={}
def norm(x):return re.sub(r'[^a-z0-9]','',x.lower().replace('miner1','').replace('miner2','').replace('miner3','').replace('obs',''))
for fid in active:
 scores=sorted([(SequenceMatcher(None,norm(fid),norm(os.path.basename(p))).ratio(),p) for p in files],reverse=True)
 score,path=scores[0]
 try:
  L=pd.read_pickle(path)
  if not isinstance(L,pd.DataFrame):raise ValueError('not dataframe')
  L=L.reindex(index=F.index,columns=A);q=pd.concat([base,L.stack().rename('y')],axis=1).dropna()
  rho=spearmanr(q.x,q.y).statistic if len(q)>=8 else np.nan
  # conservative: require recognizably matching filename, valid cells, finite correlation
  if score<.62 or len(q)<8 or not np.isfinite(rho):raise ValueError('insufficient')
  ev.append((fid,abs(float(rho)),len(q)));mapping[fid]=[os.path.basename(path),round(score,3),len(q)]
 except Exception: missing.append(fid)
mx=max(ev,key=lambda x:x[1]) if ev else None
print('LIBRARY_AUDIT',json.dumps({'active':len(active),'evidence':len(ev),'missing':missing,'max_abs_library_correlation':mx[1] if mx else None,'most_correlated':mx[0] if mx else None,'complete':not missing},sort_keys=True))
print('MAPPING',json.dumps(mapping,sort_keys=True));print('TOP_CORRELATIONS',sorted(ev,key=lambda z:-z[1])[:10])
F.to_pickle('scripts/miner_1_20350719_inverse_upside_cross_asset_beta_40obs_signal.pkl')
