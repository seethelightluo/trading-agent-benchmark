import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in symbols:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>120: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); ret=P.pct_change();
# Dispersion-conditioned reversal: invert 10D risk-adjusted move only when cross-asset dispersion is elevated.
vol=ret.rolling(30,min_periods=20).std().shift(1)
raw=-(P.pct_change(10).shift(1)/(vol*np.sqrt(252)+1e-8))
disp=ret.rolling(10,min_periods=8).std().mean(axis=1).shift(1)
threshold=disp.rolling(120,min_periods=60).median()
factor=raw.where(disp.ge(threshold), raw*0.25)
fr={h:P.shift(-h)/P-1 for h in [5,10,20,40]}
for h,y in fr.items():
  ics=[]; ns=[]; cov=[]; dates=[]
  for dt in factor.index:
   z=pd.concat([factor.loc[dt],y.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(q): ics.append(q); ns.append(len(z)); cov.append(len(z)/len(symbols)); dates.append(dt)
  a=pd.Series(ics); print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(cov):.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} start={min(dates).date()} end={max(dates).date()}')
# rank turnover and regime split at selected horizon
ranks=factor.rank(axis=1,pct=True); print('turnover',ranks.diff().abs().mean(axis=1).dropna().mean())
factor.to_csv('scripts/miner_2_20350802_dispersion_conditioned_reversal_signal.csv',index_label='date')
