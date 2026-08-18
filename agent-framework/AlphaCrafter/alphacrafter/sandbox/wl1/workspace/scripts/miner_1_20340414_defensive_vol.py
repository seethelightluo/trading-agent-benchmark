import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_account_dict
U=get_account_dict()['watch_list']; P={}
for s in U:
 d=get_stock_daily_data(s,days=5200); P[s]=d.set_index('date').close.astype(float)
px=pd.DataFrame(P).sort_index().ffill(); r=px.pct_change();
# Volatility-managed defensive carry: low realized volatility, conditioned on positive
# medium return so it avoids simply buying dormant falling assets.
vol=r.rolling(30).std()*np.sqrt(252); mom=px/px.shift(60)-1
sig=(-vol + 0.25*mom/vol.replace(0,np.nan)).shift(1)
rows=[]
for h in [5,10,20,40]:
 f=px.shift(-h)/px-1; z=[]
 for dt in sig.index:
  a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(a)>=8: z.append(a.iloc[:,0].corr(a.iloc[:,1],method='spearman'))
 print('decay',h, np.nanmean(z),len(z))
f=px.shift(-10)/px-1
for dt in sig.index:
 a=pd.concat([sig.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(a)>=8: rows.append((dt,a.iloc[:,0].corr(a.iloc[:,1],method='spearman'),len(a)))
q=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); print('dates',len(q),'avgN',q.n.mean(),'coverage',q.n.mean()/15);print('IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(),'hit',(q.ic>0).mean(),'turnover',sig.rank(axis=1,pct=True).diff().abs().stack().mean())
for a,b in [('2020','2024'),('2025','2029'),('2030','2032'),('2033','2034')]:
 z=q.loc[a:b];print(a,len(z),z.ic.mean(),z.ic.mean()/z.ic.std())
out=sig.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_1_20340414_defensive_vol_signal.csv')
