import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-05-01'); want=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in want: raw[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
px=pd.concat({s:d.close for s,d in raw.items()},axis=1).sort_index(); r=px.pct_change(); dn=r.where(r<0,0).rolling(20,min_periods=15).std().replace(0,np.nan); sig=(-px.pct_change(15)/dn).shift(1); y=px.pct_change().shift(-1)
v=[]; n=[]; tr=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  v.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); n.append(len(z)); tr.append((sig.loc[dt].rank(pct=True)-sig.shift(1).loc[dt].rank(pct=True)).abs().mean())
a=np.array(v); print(f'dates={len(a)} avgN={np.mean(n):.2f} IC={a.mean():.9f} ICIR={a.mean()/a.std(ddof=1)*np.sqrt(len(a)):.9f} hit={np.mean(a>0):.6f} turnover={np.nanmean(tr):.9f} assets={px.shape[1]} cutoff={CUT.date()}')
for start in ['2028-01-01','2029-05-01','2029-11-01','2030-01-01']:
 q=np.array([v[i] for i,dt in enumerate([x for x in sig.index if x in sig.index])]) if False else None
 vals=[]
 for dt in sig.index:
  if dt<pd.Timestamp(start): continue
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(vals); print(start,len(q),q.mean(),q.mean()/q.std(ddof=1)*np.sqrt(len(q)))
