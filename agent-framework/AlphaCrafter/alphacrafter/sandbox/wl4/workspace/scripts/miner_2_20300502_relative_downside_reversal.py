import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-05-01')
want=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in want:
  d=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index(); raw[s]=d.loc[:CUT]
px=pd.concat({s:d.close for s,d in raw.items()},axis=1).sort_index(); r=px.pct_change()
# Relative 10-session loss versus contemporaneous cross-sectional median, risk-scaled by downside volatility.
rel=r.rolling(10).sum().sub(r.rolling(10).sum().median(axis=1),axis=0)
dn=r.where(r<0,0).rolling(20,min_periods=15).std().replace(0,np.nan)
sig=(-rel/dn).shift(1)
def ev(h,start=None):
 y=px.pct_change(h).shift(-h); vals=[]; ns=[]; tr=[]
 for dt in sig.index:
  if start and dt<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   a=spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic; vals.append(a); ns.append(len(z))
   if dt in sig.index[1:]: tr.append((sig.loc[dt].rank(pct=True)-sig.shift(1).loc[dt].rank(pct=True)).abs().mean())
 a=np.array(vals); return len(a),np.mean(ns),a.mean(),a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),np.mean(a>0),np.nanmean(tr)
for h in [1,5,10,20]:
 for label,start in [('full',None),('recent252','2029-05-01'),('recent120','2029-11-01')]:
  x=ev(h,start); print(f'{label} h={h} dates={x[0]} avgN={x[1]:.2f} IC={x[2]:.6f} ICIR={x[3]:.6f} hit={x[4]:.4f} turnover={x[5]:.6f}')
print('assets',px.shape[1],'dates',px.index.min().date(),px.index.max().date())
