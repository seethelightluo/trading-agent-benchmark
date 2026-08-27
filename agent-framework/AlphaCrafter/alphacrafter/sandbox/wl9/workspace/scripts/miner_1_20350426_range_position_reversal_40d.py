import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None and len(d)>100}).sort_index()
r=px.pct_change()
# Reversal from the medium-term range: low range location implies prior selling pressure;
# combine with recent return and scale by realized risk. All inputs are lagged one session.
lo=px.rolling(40,min_periods=30).min(); hi=px.rolling(40,min_periods=30).max()
pos=(px-lo)/(hi-lo+1e-12)
recent=px.pct_change(10)
vol=r.rolling(40,min_periods=30).std()*np.sqrt(40)
f=(-(recent + 0.5*(pos-0.5))/(vol+1e-8)).shift(1)
f.to_csv('scripts/miner_1_20350426_range_position_reversal_40d_signal.csv',index_label='date')
for h in [10,20,40,60]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c); ns.append(len(z))
 x=pd.Series(vals); print(f'H={h} dates={len(x)} avgN={np.mean(ns):.2f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.6f} hit={(x>0).mean():.4f}')
print(f'coverage_valid={f.notna().sum().sum()/(f.shape[0]*len(U)):.6f} instruments={len(U)} minN=8')
rank=f.rank(axis=1,pct=True); print(f'turnover10={(rank.sub(rank.shift(10)).abs()>0.25).mean(axis=1).mean():.6f}')
for h in [10,40]:
 fr=px.shift(-h)/px-1
 for name,a,b in [('2020-22','2020','2022'),('2023-25','2023','2025'),('2026-29','2026','2029'),('2030-32','2030','2032'),('2033-35','2033','2035')]:
  vals=[]
  for dt in f.index:
   if a<=str(dt)[:4]<=b:
    z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
    if len(z)>=8:
     c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
     if pd.notna(c): vals.append(c)
  x=pd.Series(vals); print(f'h={h} regime={name} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.6f}')
