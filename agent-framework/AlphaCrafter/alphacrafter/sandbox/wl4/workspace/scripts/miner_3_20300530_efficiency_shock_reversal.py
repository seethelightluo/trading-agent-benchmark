import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-05-29')
want=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in want:
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); raw[s]=d.loc[:CUT]
px=pd.concat({s:d.close for s,d in raw.items()},axis=1).sort_index(); r=px.pct_change()
# Reversal after an inefficient, unusually large 10d shock: shock is scaled by vol;
# inefficiency is absolute net move divided by absolute path. Lag all inputs.
ret10=px.pct_change(10); vol20=r.rolling(20,min_periods=15).std(); path10=r.abs().rolling(10,min_periods=8).sum()
eff=(ret10.abs()/(path10+1e-12)).clip(0,1)
z=(ret10/(vol20+1e-12))*(1-eff)
sig=(-z).shift(1)
def ev(y,start=None):
 a=[];ns=[];tr=[]
 for dt in sig.index:
  if start and dt<pd.Timestamp(start): continue
  q=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8:
   a.append(spearmanr(q.iloc[:,0],q.iloc[:,1]).statistic);ns.append(len(q))
   tr.append((sig.loc[dt].rank(pct=True)-sig.shift(1).loc[dt].rank(pct=True)).abs().mean())
 a=np.array(a)
 return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.mean(tr)
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h)
 print('FULL h=%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%((h,)+ev(y)))
 print('RECENT h=%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%((h,)+ev(y,'2029-05-30')))
print('assets=%d range=%s..%s'%(px.shape[1],px.index.min().date(),px.index.max().date()))
