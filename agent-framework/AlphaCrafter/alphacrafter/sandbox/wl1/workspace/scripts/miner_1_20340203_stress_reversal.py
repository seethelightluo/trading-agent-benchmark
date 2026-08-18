import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 d=get_stock_daily_data(s, days=6000)
 if d is not None and len(d): D[s]=d.set_index('date')['close'].rename(s)
p=pd.concat(D,axis=1).sort_index().ffill()
r=p.pct_change()
# stress-conditioned short-horizon reversal, signal observable at t and forward return t+1:t+10
vol=r.rolling(20).std()*np.sqrt(252)
rev=-(p/p.shift(5)-1)/vol
# use cross-asset stress proxy, high VIX if available; otherwise broad realized vol percentile
stress=r.mean(axis=1).rolling(5).std().rolling(120).rank(pct=True)
# only activate top 40% stress, smoothly weighted and cross-section demeaned
f=rev.sub(rev.mean(axis=1),axis=0).mul((stress.clip(0,1)-.6).clip(lower=0),axis=0)
f=f.shift(1)
rows=[]
for i in range(len(p)-10):
 dt=p.index[i]; x=f.iloc[i]; y=p.iloc[i+10]/p.iloc[i]-1
 z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8 and z.iloc[:,0].std()>0 and z.iloc[:,1].std()>0:
  rows.append((dt,len(z),z.iloc[:,0].corr(z.iloc[:,1])))
a=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('range',p.index.min(),p.index.max(),'dates',len(a),'avgN',a.n.mean(),'coverage',a.n.mean()/15)
print('IC',a.ic.mean(),'ICIR',a.ic.mean()/a.ic.std(),'hit', (a.ic>0).mean())
for h in [(2020,2023),(2024,2026),(2027,2029),(2030,2032),(2033,2034)]:
 q=a[(a.index.year>=h[0])&(a.index.year<=h[1])];print(h,len(q),q.ic.mean(),q.ic.mean()/q.ic.std() if len(q)>1 else np.nan)
# decay
for k in [5,10,20]:
 rr=[]
 for i in range(len(p)-k):
  z=pd.concat([f.iloc[i],p.iloc[i+k]/p.iloc[i]-1],axis=1).dropna()
  if len(z)>=8: rr.append(z.iloc[:,0].corr(z.iloc[:,1]))
 print('decay',k,np.nanmean(rr),len(rr))
print('turnover',f.rank(pct=True).diff().abs().mean().mean(),'latest',f.iloc[-1].notna().sum())
# artifact
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_1_20340203_stress_reversal_signal.csv')
