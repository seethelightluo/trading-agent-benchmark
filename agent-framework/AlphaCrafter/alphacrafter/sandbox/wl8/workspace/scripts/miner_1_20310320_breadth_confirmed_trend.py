import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=get_stock_daily_data(s,days=3000)
 return d if d is not None and len(d) else get_index_daily_data(s,days=3000)
raw={s:load(s) for s in U}; close=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in raw.items()}).sort_index().ffill()
r20=close.pct_change(20); r60=close.pct_change(60); med60=r60.median(axis=1); breadth=(r60.gt(0).sum(axis=1)/r60.notna().sum(axis=1)-.5)*2
fac=r20.sub(r20.median(axis=1),axis=0)*(0.35+0.65*abs(breadth)); fac=fac.where((breadth>=0)&(med60>=0),fac.where((breadth<0)&(med60<0),fac*0.35))
fwd=close.shift(-10)/close-1; rows=[]
for dt in fac.index:
 z=pd.concat([fac.loc[dt],fwd.loc[dt]],axis=1).dropna()
 if len(z)>=8: rows.append((dt,len(z),z.iloc[:,0].rank().corr(z.iloc[:,1].rank())))
ic=pd.DataFrame(rows,columns=['date','n','ic']); ic['date']=pd.to_datetime(ic.date); ic=ic.set_index('date')
print('dates',len(ic),'start',ic.index.min(),'end',ic.index.max(),'avg_n',ic.n.mean(),'coverage',ic.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(ic.ic.mean(),ic.ic.mean()/ic.ic.std(),(ic.ic>0).mean(),fac.rank(axis=1).diff().abs().stack().mean()/14))
for name,x in [('recent180',ic.tail(180)),('recent360',ic.tail(360)),('2030',ic[ic.index.year==2030]),('2031',ic[ic.index.year==2031]),('recent60',ic.tail(60))]: print(name,len(x),'IC %.6f ICIR %.6f'%(x.ic.mean(),x.ic.mean()/x.ic.std()) if len(x) else 'empty')
for h in [1,5,10,20]:
 fw=close.shift(-h)/close-1;q=[]
 for dt in fac.index:
  z=pd.concat([fac.loc[dt],fw.loc[dt]],axis=1).dropna()
  if len(z)>=8:q.append(z.iloc[:,0].rank().corr(z.iloc[:,1].rank()))
 print('decay',h,len(q),'IC %.6f ICIR %.6f'%(np.nanmean(q),np.nanmean(q)/np.nanstd(q)))
fac.to_csv('scripts/miner_1_20310320_breadth_confirmed_trend_signal.csv');ic.to_csv('scripts/miner_1_20310320_breadth_confirmed_trend_ic.csv')
