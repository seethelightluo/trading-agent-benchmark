import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def L(s):
 for f in(get_stock_daily_data,get_index_daily_data):
  try:
   d=f(s,days=4200)
   if d is not None and len(d):return d
  except:pass
 return None
R={s:L(s) for s in U}; p=pd.DataFrame({s:(d.set_index('date').close if d is not None else pd.Series(dtype=float)) for s,d in R.items()}).sort_index().ffill()
a=pd.read_csv('../persistent/index_data/DXY.csv');a.date=pd.to_datetime(a.date);x=a.set_index('date').close.reindex(p.index).ffill()
# Dollar shock interaction: momentum is rewarded when dollar shock is benign, reversed after abnormal dollar strengthening
m=p.pct_change(20); sh=x.pct_change(5); base=sh.rolling(252,min_periods=126); g=((sh-base.mean())/base.std()).clip(-2,2).fillna(0)
s=m*(1-1.6*g.values[:,None]);s=pd.DataFrame(s,index=p.index,columns=p.columns);s=s.sub(s.median(axis=1),axis=0)
rows=[];f=p.shift(-10)/p-1
for dt in s.index:
 q=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
 if len(q)>=8:rows.append((dt,len(q),q.iloc[:,0].rank().corr(q.iloc[:,1].rank())))
i=pd.DataFrame(rows,columns=['date','n','ic']).set_index('date')
print('range',i.index.min(),i.index.max(),'dates',len(i),'avg_n',i.n.mean(),'coverage',i.n.mean()/15)
print('IC %.6f ICIR %.6f hit %.4f turnover %.6f'%(i.ic.mean(),i.ic.mean()/i.ic.std(),(i.ic>0).mean(),s.rank(axis=1,pct=True).diff().abs().stack().mean()))
for n,q in [('recent180',i.tail(180)),('recent360',i.tail(360)),('2030',i.loc['2030']),('recent60',i.tail(60))]:print(n,len(q),'IC %.6f ICIR %.6f'%(q.ic.mean(),q.ic.mean()/q.ic.std() if len(q)>1 and q.ic.std() else np.nan))
for h in[1,5,10,20]:
 z=[];f=p.shift(-h)/p-1
 for dt in s.index:
  q=pd.concat([s.loc[dt],f.loc[dt]],axis=1).dropna()
  if len(q)>=8:z.append(q.iloc[:,0].rank().corr(q.iloc[:,1].rank()))
 print('decay',h,'IC %.6f ICIR %.6f'%(np.nanmean(z),np.nanmean(z)/np.nanstd(z)))
s.to_csv('scripts/miner_3_20310417_dxy_interaction_signal.csv');i.to_csv('scripts/miner_3_20310417_dxy_interaction_ic.csv')
