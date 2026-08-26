import numpy as np,pandas as pd
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 p=pd.read_csv('../persistent/stock_data/'+s+'.csv'); p.date=pd.to_datetime(p.date); D[s]=p.set_index('date').close.astype(float)
px=pd.DataFrame(D).sort_index().ffill().loc[:'2033-12-25']; r=px.pct_change()
csmean=r.mean(axis=1); breadth=(r<0).mean(axis=1); vol=csmean.rolling(60,min_periods=30).std()
tail=(-csmean.shift(1)/vol.shift(1)).clip(0,3)
bscore=((breadth.shift(1)-0.50)/0.50).clip(0,1)
active=(breadth.shift(1)>=0.55)&tail.notna()
raw=-(px.shift(1).pct_change(5).sub(px.shift(1).pct_change(5).mean(axis=1),axis=0))
f=raw.mul((0.25+0.75*bscore)*tail,axis=0).where(active,np.nan).rank(axis=1,pct=True)
print('cutoff 2033-12-25 dates',len(px),'instruments',len(U),'coverage',round(f.notna().mean().mean(),4),'activation',round(active.mean(),4),'active_dates',int(active.sum()))
for h in [1,5,10,20]:
 vals=[]; ns=[]
 for i in range(len(px)-h):
  z=pd.concat([f.iloc[i].rename('x'),(px.iloc[i+h]/px.iloc[i]-1).rename('y')],axis=1).dropna()
  if len(z)>=8 and z.x.nunique()>2: vals.append(z.x.corr(z.y)); ns.append(len(z))
 q=pd.Series(vals).dropna(); print('H',h,'dates',len(q),'avgN',round(np.mean(ns),2),'IC',round(q.mean(),6),'ICIR',round(q.mean()/q.std(ddof=1),6),'hit',round((q>0).mean(),4))
print('turnover',round(f.diff().abs().mean().mean(),6)); f.to_csv('scripts/miner_2_20331226_breadth_tail55_signal.csv')
