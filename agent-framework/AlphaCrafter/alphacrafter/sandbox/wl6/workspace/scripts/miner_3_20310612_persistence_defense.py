import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None and len(d): px[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close
P=pd.DataFrame(px).sort_index(); r=P.pct_change()
short=r.gt(0).rolling(10,min_periods=8).mean(); long=r.gt(0).rolling(60,min_periods=40).mean()
tot=r.rolling(60,min_periods=40).std(); neg=r.clip(upper=0)
down=np.sqrt((neg**2).rolling(60,min_periods=40).mean())
defense=(1-down/(tot+1e-12)).clip(-2,2); f=(short-long)*defense
print('rows',len(P),'instruments',len(P.columns),'span',P.index.min().date(),P.index.max().date(),'nonnull',int(f.notna().sum().sum()))
for h in [5,10,20]:
 fr=P.shift(-h)/P-1; out=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   c=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(c): out.append((dt,c,len(z)))
 q=pd.DataFrame(out,columns=['date','ic','n']); q['date']=pd.to_datetime(q['date']); q=q.set_index('date'); a=q.ic.dropna()
 print('H',h,'dates',len(a),'avgN',round(q.loc[a.index,'n'].mean(),3),'coverage',round(q.loc[a.index,'n'].mean()/15,6),'IC',round(a.mean(),8),'ICIR',round(a.mean()/a.std(ddof=1)*np.sqrt(len(a)),6),'hit',round((a>0).mean(),6))
 print('years',q.loc[a.index].groupby(q.loc[a.index].index.year).ic.mean().round(4).to_dict())
print('turnover_proxy',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),6))
