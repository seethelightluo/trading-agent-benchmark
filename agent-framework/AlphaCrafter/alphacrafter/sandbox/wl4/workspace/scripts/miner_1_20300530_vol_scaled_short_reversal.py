import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-05-29')
want=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in want:
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); raw[s]=d.close.loc[:CUT]
px=pd.concat(raw,axis=1).sort_index(); r=px.pct_change()
# Short-horizon reversal scaled by recent volatility; lag signal one session.
ret=r.rolling(5,min_periods=5).sum(); vol=r.rolling(20,min_periods=15).std()
sig=(-ret/(vol+1e-12)).shift(1)
def ev(y,start=None):
 a=[]; ns=[]; tr=[]
 for dt in sig.index:
  if start is not None and dt<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
   q=sig.loc[dt].rank(pct=True); q0=sig.shift(1).loc[dt].rank(pct=True)
   tr.append((q-q0).abs().mean())
 a=np.asarray(a)
 return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.mean(tr)
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h)
 print('FULL h=%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%((h,)+ev(y)))
 print('RECENT h=%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%((h,)+ev(y,'2029-05-29')))
print('assets=%d range=%s..%s'%(px.shape[1],px.index.min().date(),px.index.max().date()))
