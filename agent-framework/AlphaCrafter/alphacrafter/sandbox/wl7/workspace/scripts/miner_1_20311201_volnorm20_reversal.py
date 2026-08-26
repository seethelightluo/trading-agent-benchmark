import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}
px=pd.DataFrame({s:d.set_index('date')['close'] for s,d in D.items() if d is not None}).sort_index().ffill()
r=np.log(px).diff()
# medium-horizon volatility-normalized reversal, lagged one completed day
vol60=r.rolling(60,min_periods=40).std()*np.sqrt(252)
f=(-px.pct_change(20).div(vol60)).shift(1)
fr=px.pct_change(10).shift(-10)
ics=[]; ns=[]; dates=[]
for dt in f.index:
 z=pd.concat([f.loc[dt],fr.loc[dt]],axis=1).dropna()
 if len(z)>=8:
  q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
  if pd.notna(q): ics.append(q); ns.append(len(z)); dates.append(dt)
a=np.array(ics)
print('span',px.index.min(),px.index.max(),'assets',len(px.columns))
print('H10 dates',len(a),'avgN',round(np.mean(ns),3),'IC',round(a.mean(),6),'ICIR',round(a.mean()/a.std(ddof=1),6),'hit',round(np.mean(a>0),4),'coverage',round(np.mean(ns)/15,4))
# regime thirds and decay
for k in [1,5,20]:
 fwd=px.pct_change(k).shift(-k); x=[]; nn=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],fwd.loc[dt]],axis=1).dropna()
  if len(z)>=8:
   q=z.iloc[:,0].corr(z.iloc[:,1],method='spearman')
   if pd.notna(q): x.append(q);nn.append(len(z))
 x=np.array(x); print('H',k,'dates',len(x),'IC',round(x.mean(),6),'ICIR',round(x.mean()/x.std(ddof=1),6),'hit',round(np.mean(x>0),4))
# signal turnover using ranks
rank=f.rank(axis=1,pct=True); turn=rank.diff().abs().mean(axis=1).dropna().mean()
print('turnover_proxy',round(float(turn),6))
pd.DataFrame({'date':dates,'ic':a,'n':ns}).to_csv('scripts/miner_1_20311201_volnorm20_reversal_ic.csv',index=False)
f.to_csv('scripts/miner_1_20311201_volnorm20_reversal_signal.csv')
