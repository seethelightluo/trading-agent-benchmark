import pandas as pd,numpy as np,glob
from scipy.stats import spearmanr
CUT=pd.Timestamp('2030-04-17')
watch=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
raw={}
for f in glob.glob('../persistent/stock_data/*.csv'):
 s=f.rsplit('/',1)[-1][:-4]
 if s in watch: raw[s]=pd.read_csv(f,parse_dates=['date']).set_index('date').sort_index().loc[:CUT]
px=pd.concat({s:raw[s].close for s in watch if s in raw},axis=1).sort_index(); r=px.pct_change()
# Downside-asymmetry momentum: medium trend divided by downside volatility,
# lagged one completed session to avoid look-ahead.
down=r.clip(upper=0).rolling(20,min_periods=15).std()
sig=(-r.rolling(15,min_periods=15).sum()/(down+1e-8)).shift(1)
y=px.pct_change().shift(-1)
a=[]; ns=[]; ranks=[]; dates=[]
for i,dt in enumerate(sig.index):
 z=pd.concat([sig.loc[dt],y.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); dates.append(dt)
  ranks.append(sig.loc[dt].rank(pct=True))
a=np.asarray(a); print('cutoff',CUT.date(),'dates',len(a),'avgN',round(np.mean(ns),2),'minN',min(ns),'assets',px.shape[1])
print('IC',a.mean(),'ICIR',a.mean()/(a.std(ddof=1)+1e-12)*np.sqrt(len(a)),'hit',np.mean(a>0),'coverage_panel',len(a)*np.mean(ns)/(len(sig)*len(watch)))
if len(ranks)>1: print('turnover',np.mean([ (ranks[i]-ranks[i-1]).abs().mean() for i in range(1,len(ranks))]))
for st in ['2020-01-01','2025-01-01','2028-01-01','2029-04-01','2029-10-01','2030-01-01']:
 q=[v for d,v in zip(dates,a) if d>=pd.Timestamp(st)]; q=np.asarray(q)
 print('regime',st,'dates',len(q),'IC',q.mean() if len(q) else np.nan,'ICIR',q.mean()/(q.std(ddof=1)+1e-12)*np.sqrt(len(q)) if len(q)>1 else np.nan)
