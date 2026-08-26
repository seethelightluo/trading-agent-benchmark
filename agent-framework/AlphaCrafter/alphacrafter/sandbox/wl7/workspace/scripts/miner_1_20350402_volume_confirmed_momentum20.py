import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,5000) for s in U}
D={s:d.set_index('date').sort_index() for s,d in D.items() if d is not None and len(d)>100}
# volume-confirmed medium momentum: 20d return times clipped relative volume, cross-sectional signal
px=pd.DataFrame({s:d['close'] for s,d in D.items()}); vol=pd.DataFrame({s:d['volume'] for s,d in D.items()})
ret=px.pct_change(); mom=px.pct_change(20); vr=(vol/vol.rolling(40,min_periods=20).median()).clip(0.5,2.0)
f=mom*vr
# forward 10 trading days
fr=px.shift(-10)/px-1
rows=[]
for dt in f.index:
 x=f.loc[dt]; y=fr.loc[dt]; z=pd.concat([x,y],axis=1).dropna()
 if len(z)>=8:
  ic=z.iloc[:,0].corr(z.iloc[:,1],method='spearman'); rows.append((dt,ic,len(z)))
r=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for n in [len(r),500,1000]:
 q=r if n==len(r) else r.tail(n)
 print('window',n,'dates',len(q),'avgN',round(q.n.mean(),2),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
print('coverage',round(f.notna().sum(axis=1).ge(8).mean(),4),'mean instruments',round(f.notna().sum(axis=1).mean(),2))
# turnover: rank signal changes, mean cross-sectional rank absolute change
ranks=f.rank(axis=1,pct=True); turnover=ranks.diff().abs().mean(axis=1).dropna()
print('turnover',round(turnover.mean(),6),'period',r.index.min(),r.index.max())
for h in [1,5,10,20]:
 yy=px.shift(-h)/px-1; a=[]
 for dt in f.index:
  z=pd.concat([f.loc[dt],yy.loc[dt]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 print('decay',h,round(np.nanmean(a),6),len(a))
# artifact
out=f.copy();out.index=out.index.strftime('%Y-%m-%d');out.to_csv('scripts/miner_1_20350402_volume_confirmed_momentum20_signal.csv')
