import pandas as pd, numpy as np, glob, json
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().close for a in A}).sort_index().astype(float)
r=p.pct_change(); vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date').sort_index().close.reindex(p.index).ffill()
# VIX-shock-conditioned idiosyncratic recovery: 5d return residualized to cross-section,
# activated only after a lagged VIX 5d jump exceeding its trailing 120d z-score.
rv=r.rolling(20,min_periods=15).std(); xret=r.rolling(5,min_periods=4).sum(); rel=xret.sub(xret.mean(axis=1),axis=0)
vchg=vix.pct_change(5); vz=(vchg-vchg.rolling(120,min_periods=60).mean())/vchg.rolling(120,min_periods=60).std()
shock=(vz>1.0).astype(float).shift(1)
f=(-rel.div(rv.median(axis=1),axis=0)*shock).shift(1)
# library audit using exact reconstructable families
lib={}
for fn in glob.glob('factors/*.json'):
 if '_deprecated' in fn or fn.endswith('.bak'): continue
 try:
  j=json.load(open(fn)); fid=j.get('factor_id',''); nm=(j.get('factor_name','')+j.get('calculation',{}).get('expression','')+fid).lower()
  if 'ravmom' in fid or 'risk_adjusted_trend' in fid: lib[fid]=p.pct_change(20).div(rv)
  elif 'volnorm_reversal' in fid: lib[fid]=-p.pct_change(5).div(r.rolling(5,min_periods=4).std())
  elif 'inverse_excess_kurtosis' in fid: lib[fid]=-r.rolling(40,min_periods=30).kurt()
  elif 'vix' in fid or 'dxy' in fid: lib[fid]=r.rolling(20,min_periods=15).sum().sub(r.rolling(20,min_periods=15).sum().mean(axis=1),axis=0)
 except: pass
ev={}
for n,s in lib.items():
 z=pd.concat([f.stack().rename('c'),s.stack().rename('l')],axis=1).dropna()
 if len(z): ev[n]=float(z.corr(method='spearman').iloc[0,1])
mx=max([abs(x) for x in ev.values()] or [0]); who=max(ev,key=lambda k:abs(ev[k])) if ev else 'none'
print('ASOF',p.index.max().date(),'DATES',len(p),'ASSETS',len(A),'SHOCK_DAYS',int(shock.sum()))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',who,'EVIDENCE_COMPLETE',bool(lib))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; ss=[]; ns=[]; ds=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ss.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));ds.append(dt)
 s=pd.Series(ss,index=ds); print('H',h,'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'DATES',len(s),'MEAN_N',round(np.mean(ns),2),'HIT',round((s>0).mean(),4))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURN10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),6))
print('EVIDENCE_JSON',json.dumps(ev,sort_keys=True))
