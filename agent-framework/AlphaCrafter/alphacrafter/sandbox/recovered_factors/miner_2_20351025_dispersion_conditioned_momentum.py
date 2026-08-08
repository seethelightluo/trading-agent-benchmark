import pandas as pd, numpy as np
from scipy.stats import spearmanr
import glob,json,os
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={a:pd.read_csv('../persistent/stock_data/'+a+'.csv').set_index('date')['close'] for a in assets}
df=pd.DataFrame(px).sort_index(); r=df.pct_change()
# Novel idea: cross-sectional dispersion-conditioned momentum. In quiet/low-dispersion
# sessions, use 20d risk-adjusted momentum; in high-dispersion sessions invert it
# (crowded trend exhaustion). All inputs are lagged before prediction.
ret20=df.pct_change(20); vol20=r.rolling(20,min_periods=15).std(); disp=r.std(axis=1).rolling(60,min_periods=40).mean(); threshold=disp.rolling(120,min_periods=80).median()
reg=(disp>threshold).astype(float)*-1 + (disp<=threshold).astype(float)
f=(ret20/vol20)*reg.shift(1); f=f.shift(1)
f=f.sub(f.mean(axis=1),axis=0)
print('rows',len(df),'assets',len(assets),'valid_cells',int(f.notna().sum().sum()),'coverage',float(f.notna().mean().mean()))
def ev(h,sub=f):
 fr=r.rolling(h).sum().shift(-h+1); xs=[]; ns=[]
 for d in sub.index:
  z=pd.concat([sub.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8: xs.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); ns.append(len(z))
 x=np.asarray(xs); return len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()),float(np.mean(ns))
for h in [1,5,10,20]: print('horizon',h,'dates IC ICIR hit meanN',ev(h))
rank=f.rank(axis=1,pct=True); print('turnover_daily_rank',float(rank.diff().abs().mean(axis=1).mean()))
for lo,hi in [('2020','2025-12-31'),('2026','2030-12-31'),('2031','2035-10-10')]: print('regime',lo,hi,'h10',ev(10,f.loc[lo:hi]))
# Exact library signal histories are not persisted for all admitted formulas.
print('library_correlation_audit','not_available_for_all_definitions')
