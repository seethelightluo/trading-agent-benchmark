import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,5000)
 if d is None or len(d)<150: d=get_index_daily_data(s,5000)
 if d is None: return None
 x=d.copy(); x.index=pd.to_datetime(x.date)
 return x[['close','volume']].astype(float)
D={s:load(s) for s in U}; D={s:x for s,x in D.items() if x is not None}
P=pd.concat({s:x.close for s,x in D.items()},axis=1).sort_index(); V=pd.concat({s:x.volume for s,x in D.items()},axis=1).reindex(P.index)
r=P.pct_change()
# Volume-shock reversal: unusually high turnover plus recent loss predicts rebound;
# normalize by 20D volatility, and lag one completed session.
volshock=(V/(V.rolling(40,min_periods=20).median())-1).clip(-2,5)
loss=(-r.rolling(5,min_periods=5).sum())
risk=r.rolling(20,min_periods=15).std().replace(0,np.nan)
f=(volshock*loss/risk).shift(1)
f.to_csv('scripts/miner_1_20350621_volume_shock_reversal_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.pct_change(h).shift(-h); vals=[]; ns=[]; turns=[]; prev=None
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if np.isfinite(c):
    vals.append(c); ns.append(len(z)); q=z.iloc[:,0].rank(pct=True)
    if prev is not None:
     ix=q.index.intersection(prev.index); turns.append(np.mean(abs(q[ix]-prev[ix])))
    prev=q
 q=np.array(vals)
 print(f'{h}D dates={len(q)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={q.mean():.8f} ICIR={q.mean()/q.std(ddof=1)*np.sqrt(len(q)):.8f} hit={np.mean(q>0):.4f} turnover={np.mean(turns):.5f}')
print('overall_dates',len(f),'avg_valid',f.notna().sum(axis=1).mean(),'cell_coverage',f.notna().sum().sum()/(len(f)*15))
