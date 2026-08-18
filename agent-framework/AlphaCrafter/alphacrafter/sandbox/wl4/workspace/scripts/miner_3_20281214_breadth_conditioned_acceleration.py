import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
P={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
p=pd.concat(P,axis=1).sort_index().loc[:'2028-12-13']; r=p.pct_change()
m20=p.shift(1)/p.shift(21)-1; m60=p.shift(1)/p.shift(61)-1
# Trend acceleration gated by the lagged cross-asset breadth regime: continuation in broad positive tape, reversal in broad negative tape.
bread=(r.rolling(20,min_periods=15).mean().gt(0).mean(axis=1).shift(1)-.5)
f=m20-m60
f=f.mul(np.sign(bread).replace(0,1),axis=0)
f=f.sub(f.mean(axis=1),axis=0)
print('candidate=breadth_conditioned_trend_acceleration cutoff=2028-12-13')
for h in [1,5,10,20]:
 a=[];ns=[];ds=[]; fr=p.pct_change(h).shift(-h)
 for t in f.index:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):a.append(c);ns.append(len(z));ds.append(t)
 q=pd.Series(a,index=ds); rr=q.tail(250)
 print(f'h={h} dates={len(q)} avgN={np.mean(ns):.2f} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f} hit={(q>0).mean():.3f} recentIC={rr.mean():.5f} recentICIR={rr.mean()/rr.std(ddof=1):.5f}')
print('coverage',round(f.notna().sum().sum()/f.size,4),'rank_turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5),'valid_dates',f.notna().any(axis=1).sum(),'assets',len(U))
