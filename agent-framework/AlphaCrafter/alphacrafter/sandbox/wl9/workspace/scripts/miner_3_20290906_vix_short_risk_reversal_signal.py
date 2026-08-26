import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y'];px={}
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is not None and len(d):
  x=d[['date','close']].copy();x.date=pd.to_datetime(x.date);px[s]=x.drop_duplicates('date').set_index('date').close.sort_index()
p=pd.DataFrame(px).sort_index().ffill();r=p.pct_change();v=r.rolling(20,min_periods=15).std()*np.sqrt(20);raw=-(p.pct_change(10)/(v+1e-8))
d=get_index_daily_data('VIX',4000);vv=d.set_index(pd.to_datetime(d.date)).close.sort_index();vv=vv.reindex(p.index).ffill();gate=vv>vv.rolling(60,min_periods=30).median();sig=raw.where(gate,0).tail(1).T;sig.columns=['signal'];sig.to_csv('scripts/miner_3_20290906_vix_short_risk_reversal_signal.csv')
