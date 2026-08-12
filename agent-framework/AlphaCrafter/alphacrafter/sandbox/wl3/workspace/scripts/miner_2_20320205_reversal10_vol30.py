import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}; close=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index()
# 10d reversal scaled by idiosyncratic volatility and smoothed with lagged 3d reversal
r=close.pct_change(); v=r.rolling(30).std(); f=(-close.pct_change(10)/v).shift(1)
fr=np.log(close.shift(-10)/close); rows=[]; ds=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ds.append(dt);ns.append(len(z))
ic=np.array(rows);ds=pd.to_datetime(ds); print('dates',len(ic),'avg_n %.2f'%np.mean(ns),'coverage %.5f'%np.mean(np.array(ns)/15));print('IC %.8f ICIR %.8f hit %.4f'%(np.mean(ic),np.mean(ic)/np.std(ic,ddof=1),np.mean(ic>0)))
for lab,lo,hi in [('2026-29','2026-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31')]:
 a=ic[(ds>=pd.Timestamp(lo))&(ds<=pd.Timestamp(hi))];print(lab,len(a),'IC %.6f ICIR %.6f'%(np.mean(a),np.mean(a)/np.std(a,ddof=1)))
print('turnover %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean());f.index.name='date';f.to_csv('scripts/miner_2_20320205_reversal10_vol30_signal.csv')
