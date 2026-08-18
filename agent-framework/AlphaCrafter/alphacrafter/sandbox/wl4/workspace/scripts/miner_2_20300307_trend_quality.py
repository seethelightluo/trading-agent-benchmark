import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-03-06')
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in watch:
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); raw[s]=d.loc[:CUT]
px=pd.concat({s:raw[s]['close'] for s in watch if s in raw},axis=1).sort_index()
r=px.pct_change(); vol=r.rolling(20,min_periods=15).std()
# Persistent trend quality: medium horizon momentum penalized by recent realized risk.
sig=(r.rolling(60,min_periods=45).sum()/(vol*np.sqrt(60))).shift(1)
fwd={h:px.pct_change(h).shift(-h) for h in [1,5,10,20]}
def obs(y, start=None, end=None):
 a=[]; ns=[]; turns=[]
 for dt in sig.index:
  if start is not None and dt<start: continue
  if end is not None and dt>end: continue
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
   q=sig.loc[dt].rank(pct=True); qp=sig.shift(1).loc[dt].rank(pct=True)
   turns.append((q-qp).abs().mean())
 a=np.asarray(a)
 return len(a),np.mean(ns),np.mean(a),np.mean(a)/(np.std(a,ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.mean(turns)
for h,y in fwd.items():
 print('h',h,'all',obs(y))
 if h==1:
  for label,st in [('2020-2024','2020-01-01'),('2025-2027','2025-01-01'),('2028-2030','2028-01-01'),('recent250','2029-03-01')]: print(label,obs(y,pd.Timestamp(st)))
print('assets',px.shape[1],'dates',px.index.min().date(),px.index.max().date())
