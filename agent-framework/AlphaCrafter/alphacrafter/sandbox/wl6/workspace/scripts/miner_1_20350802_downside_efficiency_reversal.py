import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=5000).set_index('date')['close'].astype(float) for s in syms}
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
# Lagged downside-efficiency reversal: recent return divided by downside deviation,
# with a mild liquidity-independent volatility penalty. Negative score favors losers.
down=r.where(r<0,0).rolling(30,min_periods=20).std().shift(1)
ret=r.rolling(15,min_periods=15).sum().shift(1)
vol=r.rolling(30,min_periods=20).std().shift(1)
base=-ret/(down+1e-8)
# In stressed regimes, emphasize reversal; otherwise retain a milder signal.
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
q=vix.rolling(252,min_periods=120).rank(pct=True).shift(1)
mult=(0.75+0.5*q).clip(.75,1.25)
F=base.mul(mult,axis=0)/(1+vol*10)
F=F.replace([np.inf,-np.inf],np.nan)
fr={h:P.shift(-h)/P-1 for h in [5,10,20,40]}
for h in [5,10,20,40]:
 a=[]; ns=[]; dates=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr[h].loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): a.append(c);ns.append(len(z));dates.append(d)
 a=pd.Series(a); print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} turnover={F.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean():.6f} start={min(dates).date()} end={max(dates).date()}')
F.to_csv('scripts/miner_1_20350802_downside_efficiency_reversal_signal.csv',index_label='date')
