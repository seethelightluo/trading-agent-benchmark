import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
p=pd.DataFrame({s:pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date')['close'].astype(float) for s in U}).sort_index().loc[:'2028-12-27']
r=p.pct_change(); f=-r.rolling(20,min_periods=15).std().shift(1); f=f.sub(f.mean(axis=1),axis=0)
for h in [1,5,10,20]:
 a=[]; n=[];ds=[]; fr=p.pct_change(h).shift(-h)
 for t in f.index:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c):a.append(c);n.append(len(z));ds.append(t)
 q=pd.Series(a,index=ds); rr=q.tail(250)
 print(f'h={h} dates={len(q)} avgN={np.mean(n):.2f} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} recentIC={rr.mean():.6f} recentICIR={rr.mean()/rr.std(ddof=1):.6f}')
print('coverage=',round(f.notna().sum().sum()/f.size,4),'turnover=',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
