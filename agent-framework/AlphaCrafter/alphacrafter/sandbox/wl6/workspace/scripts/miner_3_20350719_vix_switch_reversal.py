import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=5000).set_index('date')['close'].astype(float) for s in syms}
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change(); v=P.pct_change(12).shift(1)/(r.rolling(30).std().shift(1)*np.sqrt(252))
vix=pd.read_csv('../persistent/index_data/VIX.csv',parse_dates=['date']).set_index('date')['close'].reindex(P.index).ffill()
# High-VIX regime favors reversal; calm regime favors continuation. VIX threshold uses trailing 252-day percentile, lagged.
q=vix.rolling(252,min_periods=120).rank(pct=True).shift(1)
f=np.where(q.values[:,None]>=.65,-v.values,v.values); F=pd.DataFrame(f,index=P.index,columns=P.columns)
for h in [10,20,40]:
 fr=P.shift(-h)/P-1; a=[]; ns=[]; dates=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): a.append(c); ns.append(len(z)); dates.append(d)
 a=pd.Series(a); print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/len(syms):.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} start={min(dates).date()} end={max(dates).date()}')
F.to_csv('scripts/miner_3_20350719_vix_switch_reversal_signal.csv',index_label='date')
