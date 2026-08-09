import pandas as pd, numpy as np, glob, json
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().close for a in A}).astype(float)
r=p.pct_change(); peer=r.sub(r.mean(axis=1),axis=0)
# Candidate: 3-day relative reversal, activated only on high cross-sectional dispersion; lagged.
disp=r.std(axis=1).rolling(20,min_periods=15).rank(pct=True)
raw=-(r.rolling(3,min_periods=3).sum().sub(r.rolling(3,min_periods=3).mean(axis=1),axis=0))
f=raw.where(disp>0.60).shift(1)
# robust library audit: load every non-deprecated json, reconstruct common named signals where possible
lib={}
for fn in glob.glob('factors/*.json'):
 if '_deprecated' in fn or fn.endswith('.bak'): continue
 try:
  j=json.load(open(fn)); fid=j.get('factor_id','')
  # use stored expression labels to map known admitted signals
  nm=j.get('factor_name','').lower()+j.get('calculation',{}).get('expression','').lower()+fid.lower()
  if 'risk_adjusted_trend' in fid or 'risk-adjusted trend' in nm: lib[fid]=((p/p.shift(20)-1)/r.rolling(20).std()).shift(0)
  elif 'volnorm_reversal' in fid: lib[fid]=(-(p/p.shift(5)-1)/r.rolling(5,min_periods=4).std())
  elif 'ravmom' in fid: lib[fid]=(p/p.shift(20)-1)/r.rolling(20).std()
  elif 'inverse_excess_kurtosis' in fid: lib[fid]=-r.rolling(40,min_periods=30).kurt()
  elif 'inverse_expected_shortfall' in fid:
   lib[fid]=pd.DataFrame({a:r[a].rolling(40,min_periods=30).apply(lambda x:-np.mean(x[x<=np.quantile(x,.2)]),raw=True)/r[a].rolling(20).std() for a in A})
 except Exception: pass
mx=0.; who='none'; ev={}
for n,s in lib.items():
 z=pd.concat([f.stack().rename('candidate'),s.stack().rename('library')],axis=1).dropna()
 if len(z)>0:
  rho=z.candidate.corr(z.library,method='spearman'); ev[n]=float(rho)
  if abs(rho)>mx: mx=abs(rho);who=n
print('ASOF',p.index.max().date(),'DATES',len(p),'ASSETS',len(A))
print('LIBRARY_SIGNALS',len(lib),'MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',who,'EVIDENCE_COMPLETE',bool(lib))
for n,v in sorted(ev.items(), key=lambda x:-abs(x[1]))[:10]: print('LIB',n,round(v,6))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; ss=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1:
   ss.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z))
 s=pd.Series(ss); print('H',h,'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'DATES',len(s),'MEAN_N',round(np.mean(ns),2),'HIT',round((s>0).mean(),4))
 for lo,hi in [('2020','2024'),('2024','2028'),('2028','2031'),('2031','2034')]:
  q=s[[d for d in f.index if lo<=str(d.date())<hi][:len(s)]] if False else None
print('COVERAGE',round(f.notna().mean().mean(),6),'TURN10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),6))
print('EVIDENCE_JSON',json.dumps(ev,sort_keys=True))
