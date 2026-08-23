import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=5000).set_index('date')['close'].astype(float) for s in syms}
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
# Drawdown-recovery exhaustion: lagged position in the 120-session range, adjusted
# by recent realized volatility. The contrarian orientation favors assets that have
# recovered strongly toward highs only when that recovery was volatile.
lo=P.rolling(120).min(); hi=P.rolling(120).max()
range_pos=(P-lo)/(hi-lo+1e-12)
vol=r.rolling(30).std()*np.sqrt(252)
recovery=(range_pos*vol).shift(1)
F=-recovery
F.to_csv('scripts/miner_1_20350913_drawdown_recovery_60d_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.shift(-h)/P-1; cs=[]; ns=[]; dates=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):cs.append(c);ns.append(len(z));dates.append(d)
 a=pd.Series(cs)
 if len(a): print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} start={min(dates).date()} end={max(dates).date()} turnover={F.rank(axis=1,pct=True).diff().abs().mean().mean():.5f}')
