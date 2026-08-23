import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; px={}
for s in U:
 d=get_stock_daily_data(s,days=5000)
 if d is not None: px[s]=d.assign(date=pd.to_datetime(d.date)).set_index('date').close
P=pd.DataFrame(px).sort_index(); disp=P.pct_change(20).std(axis=1); high=(disp>disp.rolling(60,min_periods=30).median()).astype(float)
f=(-P.pct_change(5)).mul(high,axis=0)
def go(h):
 fr=P.shift(-h)/P-1; o=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
  if len(z)>=8:o.append((dt,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z),high.loc[dt]))
 return pd.DataFrame(o,columns=['date','ic','n','high']).set_index('date')
for h in [5,10,20]:
 q=go(h); print('H',h,'dates',len(q),'N',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
q=go(10); print('regime',q.groupby('high').ic.agg(['count','mean']));print('year',q.groupby(q.index.year).ic.agg(['count','mean']));print('coverage',q.n.mean()/15,'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
