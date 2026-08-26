import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}
close={s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None and len(d)>200}
px=pd.DataFrame(close).sort_index(); r=px.pct_change()
# Range-compression reversal: favor assets with a 60d loss, but only when
# recent realized range is compressed relative to its medium-term baseline.
ret60=px/px.shift(60)-1
rv20=r.rolling(20,min_periods=15).std()
rv120=r.rolling(120,min_periods=80).std()
compression=(rv120/(rv20+1e-8)).clip(0.25,4.0)
f=(-ret60/(rv120*np.sqrt(60)+1e-6))*compression
f=f.replace([np.inf,-np.inf],np.nan).shift(1)
f.to_csv('scripts/miner_1_20341221_range_compression_reversal_signal.csv',index_label='date')
for h in [10,20,40,60,80]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c); ns.append(len(z))
 x=pd.Series(vals).dropna(); print(f'H={h} dates={len(x)} avgN={np.mean(ns):.2f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.6f} hit={(x>0).mean():.4f}')
print(f'coverage={f.notna().sum(axis=1).mean()/len(U):.6f} instruments={len(U)}')
fr=px.shift(-60)/px-1
for name,a,b in [('2020-23','2020','2023'),('2024-26','2024','2026'),('2027-29','2027','2029'),('2030-32','2030','2032'),('2033-34','2033','2034')]:
 vals=[]
 for dt in f.index:
  if a<=str(dt)[:4]<=b:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(c): vals.append(c)
 x=pd.Series(vals).dropna(); print(f'regime={name} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.6f}')
