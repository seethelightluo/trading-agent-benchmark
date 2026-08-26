import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=cl.pct_change()
# A simple, interpretable trend-quality factor: medium-term return, rewarded only
# when cross-asset breadth confirms a healthy trend; volatility normalizes risk.
ret60=r.rolling(60,min_periods=40).sum(); ret20=r.rolling(20,min_periods=14).sum(); vol20=r.rolling(20,min_periods=14).std()
breadth=(r>0).rolling(20,min_periods=14).mean().mean(axis=1)
threshold=breadth.rolling(120,min_periods=60).quantile(.55)
confirmed=(breadth>threshold)
den=vol20*np.sqrt(20)+.02
sig=((0.65*ret60+0.35*ret20)/den).where(confirmed).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
def calc(h,mask=None):
 f=cl.shift(-h)/cl-1; vals=[]; ns=[]
 dates=np.where(mask if mask is not None else np.ones(len(sig),bool))[0]
 for i in dates:
  x=sig.iloc[i].values;y=f.iloc[i].values;ok=np.isfinite(x)&np.isfinite(y)
  if ok.sum()>=8:
   a=pd.Series(x[ok]).rank().values;b=pd.Series(y[ok]).rank().values
   if a.std()>0 and b.std()>0: vals.append(np.corrcoef(a,b)[0,1]);ns.append(ok.sum())
 z=np.asarray(vals)
 return len(z),float(z.mean()),float(z.mean()/z.std(ddof=1)),float((z>0).mean()),float(np.mean(ns))
for h in [10,20,40,60]: print('H',h,calc(h))
for n,m in [('2020-23',sig.index.year<=2023),('2024-26',sig.index.year.isin([2024,2025,2026])),('2027-29',sig.index.year.isin([2027,2028,2029])),('2030-32',sig.index.year.isin([2030,2031,2032])),('2033-34',sig.index.year>=2033)]: print(n,calc(40,m))
print('coverage',float(sig.notna().mean().mean()),'active_date_fraction',float(sig.notna().any(axis=1).mean()),'turnover',float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_1_20340817_breadth_trend_quality_signal.csv',index=False)
