import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 D[s]=x['close'].astype(float)
p=pd.concat(D,axis=1).sort_index().loc[:'2028-11-01']; r=p.pct_change()
# Prior-day close location within its trailing 20-session range; low location is reversal candidate.
lo=p.rolling(20,min_periods=15).min().shift(1); hi=p.rolling(20,min_periods=15).max().shift(1)
f=(0.5-(p.shift(1)-lo)/(hi-lo)).replace([np.inf,-np.inf],np.nan)
# reward oversold assets only when aggregate prior 5d tape is not strongly negative
bread=(r.rolling(20,min_periods=15).mean().shift(1).mean(axis=1)>-0.01).astype(float)
f=f*(0.5+0.5*bread.values[:,None])
for h in [1,5,10,20]:
 a=[];ns=[];ds=[]; fr=p.pct_change(h).shift(-h)
 for t in f.index:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8:
   a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'));ns.append(len(z));ds.append(t)
 q=pd.Series(a,index=ds).dropna(); rr=q.tail(250)
 print(f'h={h} dates={len(q)} avgN={np.mean(ns):.2f} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f} hit={(q>0).mean():.3f} recentIC={rr.mean():.5f} recentICIR={rr.mean()/rr.std(ddof=1):.5f}')
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
