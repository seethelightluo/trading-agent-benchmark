import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-04-17')
want=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in want:
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); raw[s]=d.loc[:CUT]
px=pd.concat({s:d.close for s,d in raw.items()},axis=1).sort_index()
r=px.pct_change()
# Mean-reversion score: recent 10d loss, scaled by downside volatility, with a long-term trend confirmation gate.
down=r.where(r<0,0).rolling(20,min_periods=15).std()
short=-px.pct_change(10)/(down+1e-9)
trend=px.pct_change(60)
# only favor reversal among assets not in severe long-term downtrend; smooth and lag
sig=(short * (trend>-0.15).astype(float)).shift(1)
sig=sig.rolling(3,min_periods=2).mean()
def ev(y,start=None):
 vals=[]; ns=[]; turns=[]
 for dt in sig.index:
  if start is not None and dt<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
   if dt in sig.shift(1).index:
    turns.append((sig.loc[dt].rank(pct=True)-sig.shift(1).loc[dt].rank(pct=True)).abs().mean())
 a=np.asarray(vals)
 return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.nanmean(turns)
for h in [1,5,10]:
 y=px.pct_change(h).shift(-h)
 print('h=%d full dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%(h,*ev(y)))
 print('h=%d recent252 dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%(h,*ev(y,'2029-04-17')))
print('assets=%d range=%s..%s'%(px.shape[1],px.index.min().date(),px.index.max().date()))
