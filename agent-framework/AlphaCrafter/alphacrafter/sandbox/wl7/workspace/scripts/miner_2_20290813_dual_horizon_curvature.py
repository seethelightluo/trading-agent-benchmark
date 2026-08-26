import pandas as pd, numpy as np, os
from scipy.stats import spearmanr
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
base='../persistent/stock_data'
px={}
for s in U:
 d=pd.read_csv(f'{base}/{s}.csv'); d['date']=pd.to_datetime(d['date']); px[s]=d.set_index('date')['close']
p=pd.DataFrame(px).sort_index(); r=p.pct_change()
# Dual-horizon momentum curvature: long trend minus recent trend, lagged one day.
# Positive means long trend remains strong after recent impulse fades (stable continuation).
f=(p.pct_change(60)-p.pct_change(20)).shift(1)
rows=[]
for h in [1,5,10,20]:
 ic=[]; n=[]; turnover=[]
 for dt in f.index:
  x=f.loc[dt]; y=p.pct_change(h).shift(-h).loc[dt]
  z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8:
   ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic); n.append(len(z))
 # rank turnover approximate
 rr=f.rank(axis=1,pct=True); turnover=rr.diff().abs().mean(axis=1).dropna().mean()
 a=np.asarray(ic); print(h,'dates',len(a),'avgN',np.mean(n),'IC',np.nanmean(a),'ICIR',np.nanmean(a)/np.nanstd(a,ddof=1),'hit',np.mean(a>0),'turn',turnover)
# regimes recent
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-28','2026','2028-12-31'),('2029','2029','2029-08-13')]:
 ic=[]
 for dt in f.loc[lo:hi].index:
  z=pd.concat([f.loc[dt],p.pct_change(10).shift(-10).loc[dt]],axis=1).dropna()
  if len(z)>=8: ic.append(spearmanr(z.iloc[:,0],z.iloc[:,1]).statistic)
 a=np.array(ic); print(label,len(a),np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1) if len(a)>1 else np.nan)
# save signal artifact
out=f.copy(); out.index.name='date'; out.to_csv('scripts/miner_2_20290813_dual_horizon_curvature_signal.csv')
