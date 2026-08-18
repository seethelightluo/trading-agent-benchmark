import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 D[s]=x['close'].astype(float)
p=pd.concat(D,axis=1).sort_index().loc[:'2028-11-29']; r=p.pct_change()
# Downside-adjusted momentum: trailing return divided by downside deviation,
# with all observations shifted one day to avoid lookahead.
down=r.where(r<0,0.0).rolling(30,min_periods=20).std().shift(1)
ret=p.shift(1)/p.shift(21)-1
f=(ret/(down*np.sqrt(252)+1e-8)).replace([np.inf,-np.inf],np.nan)
for h in [1,5,10,20]:
 vals=[]; ns=[]; dates=[]; fr=p.pct_change(h).shift(-h)
 for t in f.index:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8:
   vals.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman')); ns.append(len(z)); dates.append(t)
 q=pd.Series(vals,index=dates).dropna(); rr=q.tail(250)
 print(f'h={h} dates={len(q)} avgN={np.mean(ns):.2f} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f} hit={(q>0).mean():.3f} recentIC={rr.mean():.5f} recentICIR={rr.mean()/rr.std(ddof=1):.5f}')
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
