import pandas as pd, numpy as np
from scipy.stats import rankdata
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(a):
 p='../persistent/stock_data/'+a+'.csv'; return pd.read_csv(p,parse_dates=['date']).set_index('date')['close'].astype(float)
P=pd.DataFrame({a:load(a) for a in assets}).sort_index(); P=P[~P.index.duplicated(keep='last')].loc[:'2035-06-06']
V=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].astype(float).reindex(P.index).ffill()
R=P.pct_change(); r5=P.pct_change(5); r20=P.pct_change(20)
# VIX shock is observable only through prior close; fade recent relative losers more aggressively in stress.
vshock=(V/V.rolling(60,min_periods=30).median()-1).clip(-.5,2.0)
# remove common momentum component, preserving cross-sectional signal
x=r5.sub(r5.median(axis=1),axis=0)
trend=r20.sub(r20.median(axis=1),axis=0)
F=-(x-0.35*trend)* (1+0.8*vshock.shift(1)).clip(.5,2.6).values[:,None]
F=pd.DataFrame(F,index=P.index,columns=assets).clip(lower=F.quantile(.05,axis=1),upper=F.quantile(.95,axis=1),axis=0).shift(1)
def run(h):
 fr=P.pct_change(h).shift(-h); z=[]; ns=[]; ds=[]
 for dt in F.index:
  a=np.column_stack((F.loc[dt].values,fr.loc[dt].values)); ok=np.isfinite(a).all(1)
  if ok.sum()>=8:z.append(np.corrcoef(rankdata(a[ok,0]),rankdata(a[ok,1]))[0,1]);ns.append(ok.sum());ds.append(dt)
 z=np.array(z); d=pd.Index(ds); return z,np.array(ns),d
print('cutoff',P.index.max(),'rows',len(P),'assets',len(assets),'cells',int(F.notna().sum().sum()),'coverage',F.notna().sum().sum()/F.size,'turnover',F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for h in [1,5,10,20]:
 z,n,d=run(h); print('H',h,'IC %.6f ICIR %.6f dates %d meanN %.2f hit %.4f'%(z.mean(),z.mean()/z.std(ddof=1),len(z),n.mean(),(z>0).mean()))
 for lo,hi in [('2020','2024-12-31'),('2025','2029-12-31'),('2030','2032-12-31'),('2033','2035-06-06')]:
  q=(d>=lo)&(d<=hi); zz=z[q]; print(' regime',lo,'n',len(zz),'ic %.6f icir %.6f'%(zz.mean(),zz.mean()/zz.std(ddof=1)) if len(zz)>1 else ' insufficient')
print('max_abs_library_correlation unavailable; admission blocked')
