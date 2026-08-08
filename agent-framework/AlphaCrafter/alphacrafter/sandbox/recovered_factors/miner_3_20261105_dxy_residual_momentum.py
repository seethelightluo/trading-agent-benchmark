import pandas as pd, numpy as np
from scipy.stats import spearmanr
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(p):
 d=pd.read_csv(p); d.date=pd.to_datetime(d.date); return d.set_index('date').close
# align on observation dates, carrying only previously observed closes across holidays
rawpx=pd.concat([load('../persistent/stock_data/'+a+'.csv').rename(a) for a in assets],axis=1).sort_index()
dxy=load('../persistent/index_data/DXY.csv').sort_index()
idx=dxy.index
px=rawpx.reindex(idx).ffill(); dxy=dxy.reindex(idx).ffill()
r=px.pct_change(); dr=dxy.pct_change(); macro=dr.rolling(10,min_periods=8).sum()
beta=pd.DataFrame(index=idx,columns=assets,dtype=float)
for a in assets:
 q=pd.concat([r[a],dr],axis=1).dropna(); cov=q.iloc[:,0].rolling(60,min_periods=30).cov(q.iloc[:,1]); var=q.iloc[:,1].rolling(60,min_periods=30).var(); beta[a]=cov.div(var).reindex(idx)
signal=r.rolling(10,min_periods=8).sum()-beta.mul(macro,axis=0)
sig=signal.clip(signal.quantile(.1,axis=1),signal.quantile(.9,axis=1),axis=0)
print('dates',idx.min().date(),idx.max().date(),'assets',len(assets))
for h in [1,5,10,20]:
 fwd=px.pct_change(h).shift(-h); vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  ok=sig.loc[dt].notna()&fwd.loc[dt].notna()
  if ok.sum()>=8: vals.append(spearmanr(sig.loc[dt,ok],fwd.loc[dt,ok]).statistic); dates.append(dt); ns.append(ok.sum())
 z=pd.Series(vals,index=dates).dropna(); print('H',h,'dates',len(z),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.3f'%(z.mean(),z.mean()/z.std(ddof=1),(z>0).mean()))
 if h==1:
  for yr,g in z.groupby(z.index.year): print(' year',yr,'IC %.6f n %d'%(g.mean(),len(g)))
print('coverage',sig.notna().sum(axis=1).mean()/15,'turnover',sig.rank(axis=1,pct=True).diff().abs().mean().mean())
