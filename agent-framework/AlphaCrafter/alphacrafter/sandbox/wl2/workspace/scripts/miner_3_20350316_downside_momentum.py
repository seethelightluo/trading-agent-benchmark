import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 try:d=get_stock_daily_data(s,days=6000)
 except Exception:d=None
 if d is None or len(d)<150:
  try:d=get_index_daily_data(s,days=6000)
  except Exception:d=None
 return d
xs={s:get(s) for s in U}; print('loaded',[(s,0 if d is None else len(d)) for s,d in xs.items()])
px=pd.DataFrame({s:d.set_index('date').close for s,d in xs.items() if d is not None}).sort_index(); r=px.pct_change()
down=r.clip(upper=0).rolling(40,min_periods=30).std().shift(1)
sig=px.pct_change(20).shift(1)/down.replace(0,np.nan)
# cross-sectional rank signal, avoiding scale outliers
sig=sig.rank(axis=1,pct=True)
for h in [5,10,20,40]:
 fwd=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]
 for dt in sig.index:
  z=sig.loc[dt]; y=fwd.loc[dt]; ok=z.notna()&y.notna()
  if ok.sum()>=8: vals.append(z[ok].corr(y[ok],method='spearman'));dates.append(dt);ns.append(ok.sum())
 a=pd.Series(vals,index=pd.to_datetime(dates)).dropna(); recent=a[a.index>='2034-01-01']; old=a[a.index<'2034-01-01']
 print('H',h,'dates',len(a),'avgN %.2f IC %.6f ICIR %.6f hit %.4f cov %.4f recent %.6f old %.6f'%(np.mean(ns),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean(),np.mean(ns)/15,recent.mean(),old.mean()))
out=sig.stack().rename('signal').reset_index();out.columns=['date','symbol','signal'];out.to_csv('../persistent/miner_3_20350316_downside_momentum_signal.csv',index=False)
