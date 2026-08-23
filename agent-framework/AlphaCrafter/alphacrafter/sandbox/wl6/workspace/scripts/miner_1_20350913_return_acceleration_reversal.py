import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
syms=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={s:get_stock_daily_data(s,days=5000).set_index('date')['close'].astype(float) for s in syms}
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
# Interpretable acceleration reversal: recent 20-session return minus the
# average 20-session slice implied by the 120-session return, scaled by 60D vol.
# Lag the signal to ensure only completed sessions are used.
acc=P.pct_change(20)-P.pct_change(120)/6
vol=r.rolling(60).std()*np.sqrt(252)
F=(-acc/(vol+1e-8)).shift(1)
F.to_csv('scripts/miner_1_20350913_return_acceleration_reversal_signal.csv',index_label='date')
for h in [5,10,20,40]:
 fr=P.shift(-h)/P-1; cs=[];ns=[];dates=[]
 for d in F.index:
  z=pd.concat([F.loc[d],fr.loc[d]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):cs.append(c);ns.append(len(z));dates.append(d)
 a=pd.Series(cs)
 print(f'h={h} dates={len(a)} avg_n={np.mean(ns):.3f} coverage={np.mean(ns)/15:.4f} IC={a.mean():.8f} ICIR={a.mean()/a.std():.5f} hit={(a>0).mean():.4f} start={min(dates).date()} end={max(dates).date()} turnover={F.rank(axis=1,pct=True).diff().abs().mean().mean():.5f}')
