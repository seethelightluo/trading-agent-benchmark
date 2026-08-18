import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
assets=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(sym):
 d=get_stock_daily_data(sym,4500)
 if d is None or len(d)==0: d=get_index_daily_data(sym,4500)
 d=d.copy(); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').close.astype(float)
px={a:get(a) for a in assets}; prices=pd.DataFrame(px).sort_index(); r=prices.pct_change()
v=get_index_daily_data('VIX',4500); v=v.copy(); v.date=pd.to_datetime(v.date); v=v.drop_duplicates('date').set_index('date').close.astype(float).reindex(prices.index).ffill()
mom=prices/prices.shift(60)-1
# use broad realized volatility for robust coverage; downside emphasis via negative-return share
vol=r.rolling(60,min_periods=15).std()*np.sqrt(252)
downshare=(r<0).rolling(60,min_periods=15).mean()
base=mom/(vol*(0.5+downshare)+1e-8)
active=(v.shift(1)>v.shift(1).rolling(252,min_periods=126).median())
f=base.shift(1).where(active)
for h in [5,10,20]:
 fr=prices.shift(-h)/prices-1; z=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1]))
 a=pd.Series(z).dropna(); print('horizon',h,'dates',len(a),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round((a>0).mean(),4))
valid=f.notna().sum(axis=1); ad=active&(valid>=8); print('active dates',int(ad.sum()),'asset_count',len(assets),'coverage',round(valid[ad].mean()/15,4),'turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).where(ad).mean()),6))
f.loc[ad].to_csv('../persistent/miner_2_20341110_vix_trend_signal.csv',index_label='date')
