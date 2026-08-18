import pandas as pd, numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=Path('../persistent/stock_data')/(s+'.csv')
 x=pd.read_csv(p,parse_dates=['date']).set_index('date').sort_index()
 D[s]=x['close'].astype(float)
px=pd.concat(D,axis=1).sort_index().loc[:'2028-11-01']
ret=px.pct_change(); vol=ret.rolling(20,min_periods=15).std().shift(1)
# Volatility-normalized 5d pullback: negative recent return gets positive reversal score;
# cross-sectional breadth gates the reversal when the broad tape is weak.
bread=(ret.rolling(20,min_periods=15).mean().shift(1).mean(axis=1)>0).astype(float)
f=(-ret.rolling(5,min_periods=5).sum().shift(1)/vol) * (0.5+0.5*bread.values[:,None])
# Winsorize cross-section to avoid crypto outlier domination
f=f.clip(lower=f.quantile(.05,axis=1),upper=f.quantile(.95,axis=1),axis=0)
for h in [1,5,10,20]:
 vals=[]; ns=[]; dates=[]; fr=px.pct_change(h).shift(-h)
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(dt)
 q=pd.Series(vals,index=dates).dropna(); r=q.tail(250)
 print(f'h={h} dates={len(q)} avgN={np.mean(ns):.2f} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f} hit={(q>0).mean():.3f} recentIC={r.mean():.5f} recentICIR={r.mean()/r.std(ddof=1):.5f}')
print('coverage',round(np.mean([f.loc[d].notna().sum() for d in f.index])/15,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
