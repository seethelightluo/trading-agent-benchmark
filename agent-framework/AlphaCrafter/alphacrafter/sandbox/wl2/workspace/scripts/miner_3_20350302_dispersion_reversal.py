import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data, get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
    try: d=get_stock_daily_data(s,days=5000)
    except Exception: d=None
    if d is None or len(d)<150:
        try: d=get_index_daily_data(s,days=5000)
        except Exception: d=None
    return d
xs={s:get(s) for s in U}; print('loaded',[(s,0 if d is None else len(d)) for s,d in xs.items()])
rets=pd.DataFrame({s:d.set_index('date').close.pct_change() for s,d in xs.items() if d is not None}).sort_index()
px=pd.DataFrame({s:d.set_index('date').close for s,d in xs.items() if d is not None}).sort_index()
disp=rets.rolling(20).std().mean(axis=1); gate=disp.shift(1)>disp.shift(1).rolling(120).median()
sig=-(px.pct_change(10).shift(1))/rets.rolling(30).std().shift(1)
for h in [5,10,20,40]:
 fwd=px.shift(-h)/px-1; vals=[]; dates=[]; ns=[]; active=[]
 for dt in sig.index:
  if not bool(gate.get(dt,False)): continue
  z=sig.loc[dt]; ok=z.notna()&fwd.loc[dt].notna()
  if ok.sum()>=8: vals.append(z[ok].corr(fwd.loc[dt][ok],method='spearman')); dates.append(dt); ns.append(ok.sum()); active.append(True)
 a=pd.Series(vals,index=pd.to_datetime(dates)).dropna(); recent=a[a.index>='2034-01-01']; old=a[a.index<'2034-01-01']
 print('H',h,'dates',len(a),'avgN %.2f'%np.mean(ns),'IC %.6f ICIR %.6f hit %.4f cov %.4f active %.4f recent %.6f old %.6f'%(a.mean(),a.mean()/a.std(),(a>0).mean(),np.mean(ns)/15,len(a)/len(sig),recent.mean(),old.mean()))
