import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-06-12'); want=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in want:
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:CUT]; raw[s]=d
px=pd.concat({s:d.close for s,d in raw.items()},axis=1).sort_index(); vol=pd.concat({s:d.volume for s,d in raw.items()},axis=1).sort_index(); r=px.pct_change()
# Reversal strengthened when the recent move occurs on abnormally high volume.
ret=r.rolling(5,min_periods=5).sum(); vr=vol/(vol.rolling(20,min_periods=15).mean()+1e-12)
sig=(-ret*vr).shift(1)
def ev(y,start=None):
 a=[]; ns=[]; tr=[]
 for dt in sig.index:
  if start and dt<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); tr.append((sig.loc[dt].rank(pct=True)-sig.shift(1).loc[dt].rank(pct=True)).abs().mean())
 a=np.array(a); return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.mean(tr)
for h in [1,5,10,20]:
 y=px.pct_change(h).shift(-h); print('FULL h=%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%((h,)+ev(y))); print('RECENT h=%d dates=%d avgN=%.2f IC=%.6f ICIR=%.6f hit=%.4f turnover=%.6f'%((h,)+ev(y,'2029-06-12')))
print('assets=%d range=%s..%s'%(px.shape[1],px.index.min().date(),px.index.max().date()))
