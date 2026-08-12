import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}
close=pd.concat({s:d.set_index('date')['close'] for s,d in D.items() if d is not None},axis=1).sort_index()
ret20=close.pct_change(20); breadth=(ret20>0).mean(axis=1).shift(1)
trend=ret20.shift(1); rev=-close.pct_change(5).shift(1)
f=trend.where(breadth>=0.5,rev); fr=np.log(close.shift(-10)/close)
rows=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); dates.append(dt)
ic=np.array(rows); print('dates',len(ic),'avg_n',len(close.columns),'coverage',np.mean([f.loc[d].notna().sum()/15 for d in dates]))
print('IC %.8f ICIR %.8f hit %.5f'%(np.nanmean(ic),np.nanmean(ic)/np.nanstd(ic,ddof=1),np.mean(ic>0)))
for label,lo,hi in [('2020-25','2020-01-01','2025-12-31'),('2026-29','2026-01-01','2029-12-31'),('2030-32','2030-01-01','2032-01-08')]:
 a=ic[[lo<=str(d.date())<=hi for d in dates]]; print(label,len(a),'IC %.6f ICIR %.6f'%(np.nanmean(a),np.nanmean(a)/np.nanstd(a,ddof=1)) if len(a)>1 else 'NA')
print('rank_turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.index.name='date'; f.to_csv('scripts/miner_2_20320108_breadth_conditioned_trend_reversal_signal.csv')
