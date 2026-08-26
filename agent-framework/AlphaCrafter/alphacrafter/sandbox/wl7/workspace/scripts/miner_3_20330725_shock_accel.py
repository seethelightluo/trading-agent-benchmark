import numpy as np, pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p='../persistent/stock_data/'+s+'.csv'
 try:
  x=pd.read_csv(p); x.date=pd.to_datetime(x.date); D[s]=x.set_index('date').close.astype(float)
 except Exception as e: print('missing',s,e)
close=pd.DataFrame(D).sort_index().ffill(); r=close.pct_change()
shock=close.shift(1).pct_change(5)-close.shift(1).pct_change(20)/4
vol=r.shift(1).rolling(20).std(); f=-(shock/vol).replace([np.inf,-np.inf],np.nan)
print('instruments',len(D),'dates',len(close))
for h in [1,5,10,20]:
 rr=[]
 for i in range(len(close)-h):
  z=pd.concat([f.iloc[i].rename('x'),(close.iloc[i+h]/close.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2: rr.append(z.x.corr(z.y))
 q=pd.Series(rr).dropna(); print('H',h,'IC',q.mean(),'ICIR',q.mean()/q.std(ddof=1),'hit',(q>0).mean(),'n',len(q))
print('coverage',f.notna().mean().mean())
f.index=f.index.astype(str); f.to_csv('scripts/miner_3_20330725_shock_accel_signal.csv')
