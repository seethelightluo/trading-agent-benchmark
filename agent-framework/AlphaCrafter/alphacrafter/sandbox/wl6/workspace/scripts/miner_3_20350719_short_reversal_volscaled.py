import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
symbols=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in symbols:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d)>100: px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
factor=-(P.pct_change(12).shift(1) / (r.rolling(30).std().shift(1)*np.sqrt(252)))
for h in [5,10,20,40]:
 fr=P.shift(-h)/P-1; ics=[]; ns=[]; cov=[]; dates=[]
 for dt in factor.index:
  z=pd.concat([factor.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): ics.append(q); ns.append(len(z)); cov.append(len(z)/len(symbols)); dates.append(dt)
 ranks=factor.rank(axis=1,pct=True); to=ranks.diff().abs().mean(axis=1).dropna().mean(); a=pd.Series(ics)
 print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(cov):.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} turnover={to:.6f} start={min(dates).date()} end={max(dates).date()}')
factor.to_csv('scripts/miner_3_20350719_short_reversal_volscaled_signal.csv',index_label='date')
