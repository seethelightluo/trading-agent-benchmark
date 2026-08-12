import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}
close=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index()
r5=close.pct_change(5); vol=close.pct_change().rolling(20).std()
breadth=(close.pct_change(20)>0).mean(axis=1).shift(1)
mult=1.0+1.5*(0.5-breadth).clip(-0.5,0.5)
f=(-r5/vol).shift(1).mul(mult,axis=0)
fr=np.log(close.shift(-10)/close)
rows=[]; dates=[]; ns=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt); ns.append(len(z))
ic=np.asarray(rows); dates=pd.to_datetime(dates)
print('dates',len(ic),'avg_n %.2f'%np.mean(ns),'coverage %.5f'%np.mean(np.asarray(ns)/15))
print('IC %.8f ICIR %.8f hit %.5f'%(np.nanmean(ic),np.nanmean(ic)/np.nanstd(ic,ddof=1),np.mean(ic>0)))
for label,lo,hi in [('2020-25','2020-01-01','2025-12-31'),('2026-29','2026-01-01','2029-12-31'),('2030-32','2030-01-01','2032-12-31')]:
 a=ic[(dates>=pd.Timestamp(lo))&(dates<=pd.Timestamp(hi))]
 print(label,len(a),'IC %.6f ICIR %.6f hit %.4f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1),np.mean(a>0)) if len(a)>1 else 'NA')
print('rank_turnover %.6f'%f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.index.name='date'; f.to_csv('scripts/miner_2_20320205_breadth_amplified_reversal5_signal.csv')
