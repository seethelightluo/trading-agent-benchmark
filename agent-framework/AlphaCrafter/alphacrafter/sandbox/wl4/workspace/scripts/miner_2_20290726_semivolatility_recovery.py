import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=get_stock_daily_data(s,days=4000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').sort_index()
close=pd.concat({s:x.close.astype(float) for s,x in D.items()},axis=1).sort_index(); r=close.pct_change()
vol=r.rolling(20,min_periods=15).std()
down=r.clip(upper=0).rolling(30,min_periods=20).std()
# Recovery signal: favor assets with recent relative weakness, especially where downside volatility
# is elevated but the latest 5d move has begun to recover. All inputs are lagged one day.
rel5=close.pct_change(5).sub(close.pct_change(5).median(axis=1),axis=0)
rel20=close.pct_change(20).sub(close.pct_change(20).median(axis=1),axis=0)
recovery=close.pct_change(5)-close.pct_change(15).shift(5)
semivol_ratio=(down/vol).clip(0,3)
f=((-rel20/vol) + recovery/vol*semivol_ratio).shift(1).replace([np.inf,-np.inf],np.nan)
print('instruments',len(D),'dates',close.index.min().date(),close.index.max().date())
for h in [1,5,10,20]:
 fw=close.shift(-h)/close-1; vals=[]; ns=[]; cov=[]
 for dt in f.index:
  a=pd.concat([f.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(a)>=8:
   c=a.iloc[:,0].corr(a.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c); ns.append(len(a)); cov.append(len(a)/len(U))
 z=pd.Series(vals)
 print(f'h={h} dates={len(z)} avgN={np.mean(ns):.2f} IC={z.mean():.6f} ICIR={z.mean()/z.std(ddof=1):.6f} hit={(z>0).mean():.4f} coverage={np.mean(cov):.4f}')
 if h==10:
  for n in [250,500]:
   q=z.tail(n); print(f'recent{n} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f}')
print('panel_coverage',f.notna().mean().mean(),'rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean().mean())
