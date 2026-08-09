import pandas as pd, numpy as np, glob
from scipy.stats import spearmanr
base='../persistent/stock_data/'
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in syms:
 d=pd.read_csv(base+s+'.csv',parse_dates=['date']).set_index('date')['close']
 px[s]=d
p=pd.DataFrame(px).sort_index()
v=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(p.index).ffill()
# lagged signal, high-vol macro-conditioned 5d reversal
vr=v.rolling(60,min_periods=40).mean(); vs=v.rolling(60,min_periods=40).std()
high=(v>vr+1.0*vs)
ret=p.pct_change(5)
f=-ret.where(high, np.nan)
# forward close-to-close horizons
rows=[]
for h in [1,3,5,10]:
 fr=p.shift(-h)/p-1
 ics=[]; ns=[]; active=[]
 for dt in p.index:
  x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ics.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z)); active.append(dt)
 a=np.array(ics)
 print(h,'dates',len(a),'avgN',np.mean(ns),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'activefrac',len(a)/len(p.index))
print('coverage',f.notna().mean().mean(),'dates',len(p),'highfrac',high.mean())
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
