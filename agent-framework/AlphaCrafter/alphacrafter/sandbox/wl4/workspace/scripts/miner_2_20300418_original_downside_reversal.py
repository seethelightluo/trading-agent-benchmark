import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-04-17')
want=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in want:
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); raw[s]=d.loc[:CUT]
px=pd.concat({s:d.close for s,d in raw.items()},axis=1).sort_index()
ret=px.pct_change()
den=ret.where(ret<0,0.0).rolling(20,min_periods=15).std().replace(0,np.nan)
sig=(-px.pct_change(15)/den).shift(1)
def evaluate(y,start=None):
 vals=[]; ns=[]; turns=[]
 for dt in sig.index:
  if start is not None and dt<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
   turns.append((sig.loc[dt].rank(pct=True)-sig.shift(1).loc[dt].rank(pct=True)).abs().mean())
 a=np.asarray(vals)
 return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.nanmean(turns)
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h)
 for label,start in [('full',None),('recent252','2029-04-17'),('recent120','2029-10-17')]:
  r=evaluate(y,start)
  print('%s h=%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%(label,h,r[0],r[1],r[2],r[3],r[4],r[5]))
print('assets=%d data=%s..%s'%(px.shape[1],px.index.min().date(),px.index.max().date()))
