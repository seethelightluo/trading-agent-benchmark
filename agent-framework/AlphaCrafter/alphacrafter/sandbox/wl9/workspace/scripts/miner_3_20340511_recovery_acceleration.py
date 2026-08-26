import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=cl.pct_change()
vol=r.rolling(60,min_periods=40).std()*np.sqrt(252)
# Recovery acceleration: a recent rebound is scored only when it follows a materially weak medium-term path.
r20=cl.pct_change(20); r5=cl.pct_change(5)
# positive 5d recovery after 20d loss, normalized by trailing risk; symmetric zero outside setup
sig=(r5.clip(lower=0)*(-r20).clip(lower=0)/(vol+0.05)**2).clip(0,8).shift(1)
# cross-asset stress gate based on breadth of negative 20d returns, lagged with signal
breadth=(r20<0).mean(axis=1)
gate=(breadth>breadth.rolling(120,min_periods=60).quantile(.60)).astype(float)
sig=sig.mul(gate,axis=0)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
def calc(h,mask=None):
 f=cl.shift(-h)/cl-1; xs=[]; ns=[]
 dates=sig.index if mask is None else sig.index[mask]
 for dt in dates:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8:
   q=sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman')
   if pd.notna(q): xs.append(q);ns.append(ok.sum())
 x=pd.Series(xs); return len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()),float(np.mean(ns))
for h in [10,20,40,60]: print('H',h,calc(h))
for n,m in [('2020-23',sig.index.year<=2023),('2024-26',sig.index.year.isin([2024,2025,2026])),('2027-29',sig.index.year.isin([2027,2028,2029])),('2030-32',sig.index.year.isin([2030,2031,2032])),('2033-34',sig.index.year>=2033)]: print(n,calc(40,m))
print('coverage',float(sig.notna().sum(axis=1).mean()/15),'active_date_fraction',float((gate>0).mean()),'turnover',float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_3_20340511_recovery_acceleration_signal.csv',index=False)
