import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}
p=pd.concat(D,axis=1).sort_index().loc[:'2029-01-10']; r=p.pct_change()
# Short-warmup volatility-adjusted acceleration: risk-adjusted 20d trend minus 40d trend, lagged one session.
rv20=r.rolling(20,min_periods=15).std().shift(1)*np.sqrt(252)
rv40=r.rolling(40,min_periods=30).std().shift(1)*np.sqrt(252)
m20=(p.shift(1)/p.shift(21)-1)/rv20
m40=(p.shift(1)/p.shift(41)-1)/rv40
f=(m20-m40).replace([np.inf,-np.inf],np.nan)
f=f.sub(f.mean(axis=1),axis=0)
for h in [1,5,10,20]:
 a=[];ns=[];ds=[]; fr=p.pct_change(h).shift(-h)
 for t in f.index:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): a.append(c);ns.append(len(z));ds.append(t)
 q=pd.Series(a,index=ds); rr=q.tail(250)
 print(f'h={h} dates={len(q)} avgN={np.mean(ns):.2f} minN={np.min(ns)} IC={q.mean():.5f} ICIR={q.mean()/q.std(ddof=1):.5f} hit={(q>0).mean():.3f} recentIC={rr.mean():.5f} recentICIR={rr.mean()/rr.std(ddof=1):.5f}')
print('coverage',round(f.notna().sum().sum()/f.size,4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
print('cutoff',p.index.max().date(),'dates',len(p),'assets',len(U))
