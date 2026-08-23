import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 try: d=get_stock_daily_data(s, days=2700)
 except Exception: d=None
 if d is not None and len(d): px[s]=d.set_index('date')['close'].astype(float)
P=pd.DataFrame(px).sort_index().ffill(); r=P.pct_change()
trend=P.pct_change(20); vol10=r.rolling(10).std(); vol60=r.rolling(60).std()
f=((trend/(vol60+1e-12)) /(1+vol10/(vol60+1e-12))).shift(1)
for h in [1,5,10,20]:
 vals=[]
 for i in range(len(P)-h):
  x=f.iloc[i]; y=P.iloc[i+h]/P.iloc[i]-1; z=pd.concat([x,y],axis=1).dropna()
  if len(z)>=8: vals.append((P.index[i],len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
 q=pd.DataFrame(vals,columns=['date','n','ic']).dropna(); ic=q.ic.mean(); sd=q.ic.std(ddof=1)
 print(f'h={h} dates={len(q)} avg_n={q.n.mean():.2f} coverage={q.n.mean()/len(U):.4f} IC={ic:.8f} ICIR={ic/sd*np.sqrt(252):.6f} hit={(q.ic>0).mean():.4f}')
 if h==10:
  for yr,g in q.assign(year=q.date.dt.year).groupby('year'): print('year',yr,'n',len(g),'ic',round(g.ic.mean(),6))
rank=f.rank(axis=1,pct=True); basket=(rank>=.8).astype(int)
print('instruments',len(px),'dates',len(P),'turnover_proxy',(basket.diff().abs().sum(axis=1)/2).dropna().mean())
