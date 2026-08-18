import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-04-17'); want=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in want: raw[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
px=pd.concat({s:d.close for s,d in raw.items()},axis=1).sort_index(); op=pd.concat({s:d.open for s,d in raw.items()},axis=1).reindex(px.index); hi=pd.concat({s:d.high for s,d in raw.items()},axis=1).reindex(px.index); lo=pd.concat({s:d.low for s,d in raw.items()},axis=1).reindex(px.index)
base=(-(0.5*(op/px.shift(1)-1)+(px/op-1))/(((hi-lo)/px.shift(1)).rolling(20,min_periods=15).median()+1e-9)).shift(1).rolling(5,min_periods=3).mean()
r20=px.pct_change(20).shift(1); v20=px.pct_change().rolling(20,min_periods=15).std().shift(1)
# damp reversal in assets with strong own directional trend; preserve sign and cross-sectional differentiation
sig=base/(1+(r20.abs()/(v20+1e-9)).clip(0,4))
def ev(y,start=None):
 a=[];ns=[];tr=[]
 for dt in sig.index:
  if start is not None and dt<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z)); tr.append((sig.loc[dt].rank(pct=True)-sig.shift(1).loc[dt].rank(pct=True)).abs().mean())
 a=np.asarray(a); return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.mean(tr)
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h)
 print('h=%d full dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%((h,)+ev(y)))
 print('h=%d recent dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%((h,)+ev(y,'2029-04-17')))
print('assets=%d range=%s..%s'%(px.shape[1],px.index.min().date(),px.index.max().date()))
