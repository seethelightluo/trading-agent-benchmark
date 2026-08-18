import pandas as pd,numpy as np
from pathlib import Path
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=pd.read_csv(Path('../persistent/stock_data')/(s+'.csv'),parse_dates=['date']).set_index('date').sort_index()
 D[s]=x['close'].astype(float)
p=pd.DataFrame(D).sort_index().loc[:'2028-12-27']; r=p.pct_change(); mkt=r.median(axis=1)
res=r.sub(mkt,axis=0)
vol=res.rolling(30,min_periods=20).std().shift(1)*np.sqrt(252)
raw=(p.shift(1)/p.shift(21)-1) - mkt.rolling(20,min_periods=15).sum().shift(1).to_numpy()[:,None]
f=(raw/vol).replace([np.inf,-np.inf],np.nan)
f=f.sub(f.mean(axis=1),axis=0)
print('universe=15; cutoff=2028-12-27')
for h in [1,5,10,20]:
 vals=[]; ns=[]; dates=[]; fr=p.pct_change(h).shift(-h)
 for t in f.index:
  z=pd.concat([f.loc[t],fr.loc[t]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): vals.append(c);ns.append(len(z));dates.append(t)
 q=pd.Series(vals,index=dates); recent=q.tail(250)
 print(f'h={h} dates={len(q)} avgN={np.mean(ns):.2f} IC={q.mean():.6f} ICIR={q.mean()/q.std(ddof=1):.6f} hit={(q>0).mean():.3f} recent250IC={recent.mean():.6f} recent250ICIR={recent.mean()/recent.std(ddof=1):.6f}')
print('coverage=',round(f.notna().sum().sum()/f.size,4),'turnover=',round(f.rank(axis=1,pct=True).diff().abs().mean().mean(),5))
