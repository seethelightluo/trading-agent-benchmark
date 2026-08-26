import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None and len(d)>200}).sort_index()
r=px.pct_change()
# Recovery quality: assets that recovered a large fraction of their 60d peak-to-trough drawdown
rollmax=px.rolling(60,min_periods=45).max(); dd=px/rollmax-1
trough=dd.rolling(60,min_periods=45).min().abs()
rec=(px/px.rolling(60,min_periods=45).min()-1)
# favor recovery from deep drawdowns, but penalize unstable downside volatility
negvol=r.clip(upper=0).abs().rolling(40,min_periods=30).std()
f=(rec/(trough+1e-5))/(negvol+1e-5)
f=f.replace([np.inf,-np.inf],np.nan).shift(1)
f.to_csv('scripts/miner_2_20350201_drawdown_recovery_quality_signal.csv',index_label='date')
for h in [10,20,40,60]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c); ns.append(len(z))
 x=pd.Series(vals); print(f'H={h} dates={len(x)} avgN={np.mean(ns):.2f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.6f} hit={(x>0).mean():.4f}')
print(f'coverage={f.notna().sum(axis=1).mean()/len(U):.6f} instruments={len(U)}')
fr=px.shift(-60)/px-1
for name,a,b in [('2020-23','2020','2023'),('2024-26','2024','2026'),('2027-29','2027','2029'),('2030-32','2030','2032'),('2033-35','2033','2035')]:
 vals=[]
 for dt in f.index:
  if a<=str(dt)[:4]<=b:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(c): vals.append(c)
 x=pd.Series(vals); print(f'regime={name} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.6f}')
q=f.rank(axis=1,pct=True); turn=[]
for i in range(1,len(q)):
 z=pd.concat([q.iloc[i],q.iloc[i-1]],axis=1).dropna()
 if len(z)>=8: turn.append((z.iloc[:,0]-z.iloc[:,1]).abs().mean())
print(f'rank_turnover={np.mean(turn):.6f} turnover_dates={len(turn)}')
