import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=6000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'].astype(float) for s,d in D.items() if d is not None and len(d)>220}).sort_index()
r=px.pct_change(); r20=px.pct_change(20); r60=px.pct_change(60); r120=px.pct_change(120)
vol=r.rolling(60,min_periods=45).std()*np.sqrt(60)
agree=((np.sign(r20)==np.sign(r60)).astype(float)+(np.sign(r60)==np.sign(r120)).astype(float))/2
# Reversal score: reverse risk-adjusted 60D trend, conditional on trend agreement.
f=(-r60/(vol+1e-8)*agree).shift(1)
f.to_csv('scripts/miner_2_20350315_trend_agreement_60d_signal.csv',index_label='date')
for h in [10,20,40,60]:
 fr=px.shift(-h)/px-1; vals=[]; ns=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c); ns.append(len(z))
 x=pd.Series(vals); print(f'H={h} dates={len(x)} avgN={np.mean(ns):.2f} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.6f} hit={(x>0).mean():.4f}')
print(f'coverage={f.notna().sum(axis=1).mean()/len(U):.6f} instruments={len(U)}')
rank=f.rank(axis=1,pct=True); print(f'turnover10={((rank.sub(rank.shift(10)).abs()>0.25).mean(axis=1)).mean():.6f}')
fr=px.shift(-40)/px-1
for name,a,b in [('2024-26','2024','2026'),('2027-29','2027','2029'),('2030-32','2030','2032'),('2033-35','2033','2035')]:
 vals=[]
 for dt in f.index:
  if a<=str(dt)[:4]<=b:
   z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
   if len(z)>=8:
    c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
    if pd.notna(c): vals.append(c)
 x=pd.Series(vals); print(f'regime={name} dates={len(x)} IC={x.mean():.6f} ICIR={x.mean()/x.std(ddof=1)*np.sqrt(252):.6f} hit={(x>0).mean():.4f}')