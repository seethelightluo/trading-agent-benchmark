import pandas as pd, numpy as np, glob, json
from scipy.stats import spearmanr
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({a:pd.read_csv('../persistent/stock_data/'+a+'.csv',parse_dates=['date']).set_index('date').sort_index().close for a in A}).astype(float).sort_index()
r=p.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Volatility-normalized relative 10-day reversal, using only completed data and one-day lag.
ret10=p.pct_change(10); crossvol=vol.median(axis=1)
f=(-(ret10-ret10.mean(axis=1))/crossvol).shift(1)
lib={}; ev={}
for fn in glob.glob('factors/*.json'):
 if '_deprecated' in fn or fn.endswith('.bak'): continue
 try:
  j=json.load(open(fn)); fid=j.get('factor_id',''); nm=(j.get('factor_name','')+j.get('calculation',{}).get('expression','')+fid).lower()
  if 'risk_adjusted_trend' in fid or 'risk-adjusted trend' in nm: lib[fid]=(p.pct_change(20)/r.rolling(20).std())
  elif 'volnorm_reversal' in fid: lib[fid]=(-p.pct_change(5)/r.rolling(5,min_periods=4).std())
  elif 'ravmom' in fid: lib[fid]=p.pct_change(20)/r.rolling(20).std()
  elif 'inverse_excess_kurtosis' in fid: lib[fid]=-r.rolling(40,min_periods=30).kurt()
  elif 'inverse_expected_shortfall' in fid: lib[fid]=pd.DataFrame({a:r[a].rolling(40,min_periods=30).apply(lambda x:-np.mean(x[x<=np.quantile(x,.2)]),raw=True)/r[a].rolling(20).std() for a in A})
 except Exception: pass
for n,s in lib.items():
 z=pd.concat([f.stack().rename('c'),s.stack().rename('l')],axis=1).dropna()
 if len(z): ev[n]=float(z.corr(method='spearman').iloc[0,1])
mx=max([abs(x) for x in ev.values()] or [0]); who=max(ev,key=lambda k:abs(ev[k])) if ev else 'none'
print('ASOF',p.index.max().date(),'DATES',len(p),'ASSETS',len(A),'LIBRARY_SIGNALS',len(lib))
print('MAX_ABS_LIBRARY_CORRELATION',round(mx,6),'FACTOR',who,'EVIDENCE_COMPLETE',bool(lib))
for h in [1,5,10,20]:
 y=p.shift(-h)/p-1; ss=[]; ns=[]; dates=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8 and z.iloc[:,0].nunique()>1 and z.iloc[:,1].nunique()>1:
   ss.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));dates.append(dt)
 s=pd.Series(ss,index=dates); print('H',h,'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'DATES',len(s),'MEAN_N',round(np.mean(ns),2),'HIT',round((s>0).mean(),4))
 for lo,hi in [('2020','2024'),('2024','2028'),('2028','2031'),('2031','2034')]:
  q=s[(s.index.astype(str)>=lo)&(s.index.astype(str)<hi)]
  if len(q): print(' REGIME',lo,hi,'IC',round(q.mean(),6),'N',len(q))
print('COVERAGE',round(f.notna().mean().mean(),6),'TURN10',round(f.rank(axis=1,pct=True).diff(10).abs().mean(axis=1).mean(),6))
print('EVIDENCE_JSON',json.dumps(ev,sort_keys=True))
