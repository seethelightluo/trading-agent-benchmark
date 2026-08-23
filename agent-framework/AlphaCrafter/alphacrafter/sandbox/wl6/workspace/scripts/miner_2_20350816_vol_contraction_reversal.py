import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in symbols:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
# Volatility contraction reversal: recent selloff is favored when current vol is below its long-run level.
# Every component is shifted so only completed sessions are used.
rv10=r.rolling(10,min_periods=8).std().shift(1)
rv40=r.rolling(40,min_periods=30).std().shift(1)
contraction=(1-rv10/(rv40+1e-12)).clip(-2,2)
reversal=-(P.pct_change(10).shift(1))
factor=(reversal/(rv40+1e-12))*contraction
factor=factor.replace([np.inf,-np.inf],np.nan)
for h in [5,10,20,40]:
 y=P.shift(-h)/P-1; ics=[]; ns=[]; dates=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): ics.append(q); ns.append(len(z)); dates.append(dt)
 a=pd.Series(ics); print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(np.array(ns)/15):.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} start={min(dates).date()} end={max(dates).date()}')
# subperiod and signal turnover
ranks=factor.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).dropna().mean())
for name,lo,hi in [('early','2020-01-01','2025-12-31'),('mid','2026-01-01','2030-12-31'),('recent','2031-01-01','2035-07-31')]:
 vals=[]; y=P.shift(-10)/P-1
 for dt in factor.loc[lo:hi].index:
  z=pd.concat([factor.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(z)>=8: vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(vals).dropna(); print('regime',name,'dates',len(a),'IC',a.mean(),'ICIR',a.mean()/a.std() if a.std()>0 else np.nan)
factor.to_csv('scripts/miner_2_20350816_vol_contraction_reversal_signal.csv',index_label='date')
