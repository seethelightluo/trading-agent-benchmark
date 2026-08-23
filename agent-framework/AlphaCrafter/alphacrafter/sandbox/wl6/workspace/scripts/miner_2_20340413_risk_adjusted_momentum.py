import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}
close=pd.DataFrame({s:(d.set_index('date')['close'] if d is not None else pd.Series(dtype=float)) for s,d in D.items()}).sort_index()
ret=close.pct_change()
f=(close.shift(5)/close.shift(65)-1)/(ret.rolling(30).std().shift(5)+1e-12)
for h in [5,10,20,40]:
  ics=[]; ns=[]
  for dt in f.index:
    z=pd.concat([f.loc[dt],close.pct_change(h).shift(-h).loc[dt]],axis=1).dropna()
    if len(z)>=8:
      q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
      if np.isfinite(q): ics.append(q); ns.append(len(z))
  a=np.array(ics)
  print(h,'dates',len(a),'avg_n',round(np.mean(ns),3),'coverage',round(np.mean(ns)/15,4),'IC',round(a.mean(),7),'ICIR',round(a.mean()/a.std(),7),'hit',round(np.mean(a>0),4))
r=f.rank(axis=1,pct=True); turn=r.diff().abs().mean(axis=1).dropna()
print('turnover_proxy',round(float(turn.mean()),6),'factor_dates',int(f.notna().sum(axis=1).ge(8).sum()),'last',str(close.index.max().date()))
