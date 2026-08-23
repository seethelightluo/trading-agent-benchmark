import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
    x=get_stock_daily_data(s, days=5000)
    if x is not None and len(x):
        x=x.copy(); x['date']=pd.to_datetime(x['date']); D[s]=x.set_index('date')['close'].astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
vol=r.rolling(30,min_periods=15).std()*np.sqrt(252)
sig=(p/p.shift(20)-1)/vol
breadth=(r.rolling(20).sum()>0).mean(axis=1)
sig=sig.mul((0.5+breadth),axis=0)
print('dates',p.index.min(),p.index.max(),'assets',len(p.columns),'rows',len(p))
for h in [5,10,20]:
  ics=[]; dates=[]; ns=[]; turnovers=[]; prev=None
  for i in range(len(p)-h):
    f=sig.iloc[i]; y=p.iloc[i+h]/p.iloc[i]-1; z=pd.concat([f,y],axis=1).dropna()
    if len(z)>=8:
      ic=z.iloc[:,0].rank().corr(z.iloc[:,1].rank())
      if np.isfinite(ic): ics.append(ic); dates.append(p.index[i]); ns.append(len(z))
    rr=f.dropna().rank()
    if prev is not None:
      q=pd.concat([rr,prev],axis=1).dropna()
      if len(q): turnovers.append((q.iloc[:,0]!=q.iloc[:,1]).mean())
    prev=rr
  a=np.array(ics); sd=a.std(ddof=1)
  print('H',h,'dates',len(a),'avgN',round(np.mean(ns),3),'IC',round(a.mean(),8),'ICIR',round(a.mean()/sd*np.sqrt(252),5),'hit',round((a>0).mean(),5),'turnover',round(np.mean(turnovers),5))
  if h==10: ten=(a,dates)
print('regimes10')
for yr,g in pd.Series(ten[0],index=ten[1]).groupby(lambda x:x.year): print(yr,len(g),round(g.mean(),6))
print('coverage',round(sig.notna().sum(axis=1).mean()/len(U),5))
