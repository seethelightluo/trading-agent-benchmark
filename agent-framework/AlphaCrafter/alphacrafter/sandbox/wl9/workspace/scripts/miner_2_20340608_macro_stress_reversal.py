import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5200) for s in U}
cl=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=cl.pct_change()
# Observation-only macro stress: elevated VIX or DXY 20d appreciation.
def macro(name):
 p='../persistent/index_data/'+name+'.csv'; x=pd.read_csv(p); x['date']=pd.to_datetime(x['date']); x=x.set_index('date')['close']; return x.reindex(cl.index).ffill()
vix=macro('VIX'); dxy=macro('DXY')
stress=((vix>vix.rolling(252,min_periods=120).quantile(.70)) | (dxy.pct_change(20)>dxy.pct_change(20).rolling(252,min_periods=120).quantile(.75))).astype(float)
ret10=cl.pct_change(10)
vol20=r.rolling(20,min_periods=15).std()*np.sqrt(252)
# Stress-conditioned downside shock rebound, normalized by recent volatility.
sig=(-ret10.clip(upper=0)/(vol20+0.05)).clip(0,8).mul(stress,axis=0).shift(1)
print('period',cl.index.min().date(),cl.index.max().date(),'assets',len(cl.columns),'stress_frac',stress.mean())
def calc(h, mask=None):
 f=cl.shift(-h)/cl-1; xs=[]; ns=[]
 for dt in sig.index if mask is None else sig.index[mask]:
  ok=sig.loc[dt].notna()&f.loc[dt].notna()
  if ok.sum()>=8 and sig.loc[dt,ok].nunique()>1:
   q=sig.loc[dt,ok].corr(f.loc[dt,ok],method='spearman')
   if pd.notna(q): xs.append(q);ns.append(ok.sum())
 x=pd.Series(xs); return len(x),x.mean(),x.mean()/x.std(ddof=1),(x>0).mean(),np.mean(ns)
for h in [10,20,40,60]: print('H',h,calc(h))
for n,m in [('2020-23',sig.index.year<=2023),('2024-26',sig.index.year.isin([2024,2025,2026])),('2027-29',sig.index.year.isin([2027,2028,2029])),('2030-32',sig.index.year.isin([2030,2031,2032])),('2033-34',sig.index.year>=2033)]: print(n,calc(60,m))
print('coverage',sig.notna().sum(axis=1).mean()/15,'active_date_fraction',float((stress>0).mean()),'turnover',sig.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
sig.stack().rename('signal').reset_index().set_axis(['date','symbol','signal'],axis=1).to_csv('scripts/miner_2_20340608_macro_stress_reversal_signal.csv',index=False)
