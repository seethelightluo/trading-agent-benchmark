import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}
close={s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None and len(d)>200}
px=pd.DataFrame(close).sort_index(); r=px.pct_change()
# Relative reversal: fade each asset's 60D return versus contemporaneous cross-asset median,
# scaled by its 60D idiosyncratic (market-relative) volatility.
rel=r.sub(r.median(axis=1),axis=0)
rv=rel.rolling(60,min_periods=45).std()*np.sqrt(60)
asset_ret=px/px.shift(60)-1
market_ret=asset_ret.median(axis=1)
f=-(asset_ret.sub(market_ret,axis=0))/(rv+1e-5)
f=f.replace([np.inf,-np.inf],np.nan).shift(1)
f.to_csv('scripts/miner_3_20350118_relative_reversal_signal.csv',index_label='date')
for h in [10,20,40,60,80]:
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
 x=pd.Series(vals); print(f'regime={name} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.6f} hit={(x>0).mean():.4f}')