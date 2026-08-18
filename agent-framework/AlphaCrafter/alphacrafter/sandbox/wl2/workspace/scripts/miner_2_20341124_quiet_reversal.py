import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data, get_index_daily_data
A=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,4500)
 if d is None or len(d)==0: d=get_index_daily_data(s,4500)
 d=d.copy(); d.date=pd.to_datetime(d.date); return d.drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame({s:load(s) for s in A}).sort_index(); r=p.pct_change()
vol=r.rolling(20,min_periods=10).std();
# Contrarian short-horizon loss, normalized by ex-ante volatility and gated to quiet relative regime
rv=vol/(r.rolling(120,min_periods=40).mean().abs()+vol.rolling(120,min_periods=40).median()+1e-8)
quiet=rv.shift(1).rank(axis=1,pct=True).mean(axis=1)<0.60
f=(-r.rolling(3,min_periods=3).sum()/(vol*np.sqrt(3)+1e-8)).shift(1).where(quiet)
for h in [3,5,10,20]:
 fr=p.shift(-h)/p-1; z=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].corr(q.iloc[:,1]))
 s=pd.Series(z).dropna(); print('horizon',h,'dates',len(s),'IC',round(s.mean(),6),'ICIR',round(s.mean()/s.std(ddof=1),6),'hit',round((s>0).mean(),4))
valid=f.notna().sum(axis=1); use=quiet&(valid>=8); print('active_dates',int(use.sum()),'assets',len(A),'coverage',round(valid[use].mean()/15,4),'turnover',round(float(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).where(use).mean()),6))
f.loc[use].to_csv('../persistent/miner_2_20341124_quiet_reversal_signal.csv',index_label='date')
