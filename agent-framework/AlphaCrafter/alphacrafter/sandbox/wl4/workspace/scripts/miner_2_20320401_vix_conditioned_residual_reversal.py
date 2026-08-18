import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=f'../persistent/stock_data/{s}.csv'
 if os.path.exists(p):
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
px=pd.concat(D,axis=1).sort_index().loc[:'2032-03-31']
macro=pd.read_csv('../persistent/index_data/VIX.csv'); macro.date=pd.to_datetime(macro.date)
vix=macro.set_index('date').close.reindex(px.index).ffill()
lr=np.log(px).diff(); mkt=lr.mean(axis=1)
# One interpretable idea: residual 20d reversal, scaled by asset vol, intensified when VIX is rising.
res20=(lr.rolling(20).sum().sub(mkt.rolling(20).sum(),axis=0))
vol20=lr.rolling(20,min_periods=15).std()
vixshock=(np.log(vix).diff(10).clip(lower=0)).fillna(0)
f=(-res20.div(vol20)).mul(1+vixshock,axis=0).shift(1)
for h in [5,10,20]:
 fr=np.log(px.shift(-h)/px); a=[]; ns=[]; turns=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8: a.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
  r=f.loc[dt].rank(pct=True)
  if prev is not None:
   q=pd.concat([r,prev],axis=1).dropna(); turns.append((q.iloc[:,0]-q.iloc[:,1]).abs().mean())
  prev=r
 a=np.asarray(a); ic=np.nanmean(a); ir=ic/(np.nanstd(a,ddof=1)/np.sqrt(len(a)))
 print(f'H{h} dates={len(a)} avgN={np.mean(ns):.2f} IC={ic:.6f} ICIR={ir:.4f} hit={np.mean(a>0):.4f} turnover={np.nanmean(turns):.4f}')
 if h==10:
  for n in [260,520,780]:
   q=a[-n:]; print(f'recent{n} IC={np.mean(q):.6f} ICIR={np.mean(q)/(np.std(q,ddof=1)/np.sqrt(len(q))):.4f} hit={np.mean(q>0):.4f}')
print('cutoff',px.index.max().date(),'assets',px.shape[1],'vix_coverage',vix.notna().mean())
