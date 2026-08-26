import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill(); r=cl.pct_change()
cs60=r.rolling(60,min_periods=40).sum(); residual=cs60.sub(cs60.median(axis=1),axis=0)
rec10=r.rolling(10,min_periods=7).sum(); vol20=r.rolling(20,min_periods=12).std()
disp=cs60.std(axis=1); active=disp > disp.rolling(120,min_periods=60).quantile(.70)
den=vol20*np.sqrt(20)+.02
sig=((-residual/den)+1.5*rec10/den).where(active).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns))
def calc(h,mask=None):
 f=cl.shift(-h)/cl-1; vals=[]; ns=[]
 for i in np.where(mask if mask is not None else np.ones(len(sig),bool))[0]:
  x=sig.iloc[i].values; y=f.iloc[i].values; ok=np.isfinite(x)&np.isfinite(y)
  if ok.sum()>=8:
   a=pd.Series(x[ok]).rank().values; b=pd.Series(y[ok]).rank().values
   if a.std()>0 and b.std()>0: vals.append(np.corrcoef(a,b)[0,1]); ns.append(ok.sum())
 x=np.asarray(vals); return len(x),float(x.mean()),float(x.mean()/x.std(ddof=1)),float((x>0).mean()),float(np.mean(ns))
for h in [10,20,40,60]: print('H',h,calc(h))
for n,m in [('2020-23',sig.index.year<=2023),('2024-26',sig.index.year.isin([2024,2025,2026])),('2027-29',sig.index.year.isin([2027,2028,2029])),('2030-32',sig.index.year.isin([2030,2031,2032])),('2033-34',sig.index.year>=2033)]: print(n,calc(40,m))
print('coverage',float(sig.notna().sum(axis=1).mean()/15),'active',float(sig.notna().mean().mean()),'turnover',float(sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean()))
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_1_20340914_dispersion70_residual_recovery_signal.csv',index=False)
