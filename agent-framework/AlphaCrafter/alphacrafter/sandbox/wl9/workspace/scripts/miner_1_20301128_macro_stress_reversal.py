import os, numpy as np, pandas as pd
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
def load(sym):
 d=pd.read_csv(os.path.join(base,sym+'.csv'),parse_dates=['date']).sort_values('date').set_index('date')
 return d['close'].astype(float)
px=pd.concat({s:load(s) for s in U},axis=1)
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).sort_values('date').set_index('date')['close'].astype(float)
# Factor: lagged, volatility-scaled short reversal, amplified when VIX is above its 60d median.
r=px.pct_change(10); vol=px.pct_change().rolling(20).std()*np.sqrt(252)
stress=(vix/vix.rolling(60).median()-1).clip(lower=0,upper=2).reindex(px.index).ffill()
f=(-(r/vol)*(1+stress)).shift(1)
for h in [5,10,20,40,60]:
 fr=px.pct_change(h).shift(-h)
 vals=[]; ns=[]; dates=[]
 for dt in f.index:
  a=f.loc[dt]; b=fr.loc[dt]; ok=a.notna()&b.notna()
  if ok.sum()>=8:
   vals.append(spearmanr(a[ok],b[ok]).statistic); ns.append(ok.sum()); dates.append(dt)
 x=np.array(vals); print(h,'dates',len(x),'avgN',round(np.mean(ns),2),'IC',round(np.nanmean(x),6),'ICIR',round(np.nanmean(x)/np.nanstd(x,ddof=1),6),'hit',round(np.mean(x>0),4))
# turnover rank proxy and coverage
rank=f.rank(axis=1,pct=True); turn=(rank.diff().abs().mean(axis=1)/2).mean(); cov=f.notna().sum(axis=1).mean()/15
print('turnover',round(turn,6),'coverage',round(cov,6),'period',f.index.min().date(),f.index.max().date())
