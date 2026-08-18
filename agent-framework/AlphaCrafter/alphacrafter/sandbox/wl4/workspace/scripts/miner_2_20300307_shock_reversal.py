import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-03-06'); watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in watch: raw[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
px=pd.concat({s:raw[s].close for s in watch if s in raw},axis=1).sort_index(); r=px.pct_change(); v=r.rolling(20,min_periods=15).std()
# One-day mean-reversion shock, normalized by recent volatility and gated by an elevated-range condition.
shock=(-r/v).shift(1); tr=None
op=pd.concat({s:raw[s].open for s in raw},axis=1).reindex(px.index); hi=pd.concat({s:raw[s].high for s in raw},axis=1).reindex(px.index); lo=pd.concat({s:raw[s].low for s in raw},axis=1).reindex(px.index)
tr=(hi-lo)/px.shift(1); gate=(tr.rolling(3,min_periods=3).mean()/tr.rolling(60,min_periods=40).median()).clip(.5,3)
sig=shock*gate; y=px.pct_change().shift(-1); a=[];ns=[];turn=[]
for dt in sig.index:
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic);ns.append(len(z));turn.append((sig.loc[dt].rank(pct=True)-sig.shift(1).loc[dt].rank(pct=True)).abs().mean())
a=np.array(a); print('dates',len(a),'avgN',np.mean(ns),'IC',np.mean(a),'ICIR',np.mean(a)/(np.std(a,ddof=1)+1e-12)*np.sqrt(len(a)),'hit',np.mean(a>0),'turnover',np.mean(turn),'assets',px.shape[1])
for st in ['2020-01-01','2025-01-01','2028-01-01','2029-03-01']:
 q=[]
 for dt in sig.index[sig.index>=pd.Timestamp(st)]:
  z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 q=np.array(q);print(st,len(q),q.mean(),q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(len(q)))
