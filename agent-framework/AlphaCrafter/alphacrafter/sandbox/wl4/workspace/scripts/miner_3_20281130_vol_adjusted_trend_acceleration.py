import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 D[s]=x['close'].astype(float)
p=pd.concat(D,axis=1).sort_index().loc[:'2028-11-29']; r=p.pct_change()
# Volatility-adjusted trend acceleration: recent risk-adjusted return relative to slower risk-adjusted trend.
# Every input is shifted one completed session before forming the signal.
rv20=r.rolling(20,min_periods=15).std().shift(1)*np.sqrt(252)
rv60=r.rolling(60,min_periods=40).std().shift(1)*np.sqrt(252)
m20=(p.shift(1)/p.shift(21)-1)/rv20
m60=(p.shift(1)/p.shift(61)-1)/rv60
f=(m20-m60).replace([np.inf,-np.inf],np.nan)
# cross-sectional demeaning makes the signal relative across the 15 tradable assets
f=f.sub(f.mean(axis=1),axis=0)
for h in [1,5,10,20]:
 a=[];ns=[];ds=[]; fr=p.pct_change(h).shift(-h)
 for t in f.index:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): a.append(c); ns.append(len(z)); ds.append(t)
 q=pd.Series(a,index=ds); rr=q.tail(250)
 print(f'h={h} dates={len(q)} avgN={np.mean(ns):.2f} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f} hit={(q>0).mean():.3f} recentIC={rr.mean():.5f} recentICIR={rr.mean()/rr.std(ddof=1):.5f}')
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
