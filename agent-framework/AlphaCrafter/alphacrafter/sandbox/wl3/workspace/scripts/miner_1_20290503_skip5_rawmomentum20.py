import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None or len(d)==0:continue
 d=d.sort_values('date'); c=d.close.astype(float)
 x=pd.DataFrame({'date':pd.to_datetime(d.date),'symbol':s,'mom':c.shift(5)/c.shift(25)-1,'fwd5':c.shift(-5)/c-1})
 rows.append(x)
a=pd.concat(rows,ignore_index=True); a['factor_value']=a.mom-a.groupby('date').mom.transform('median')
out=[]
for dt,g in a.groupby('date'):
 g=g.dropna(subset=['factor_value','fwd5'])
 if len(g)>=8 and g.factor_value.nunique()>1 and g.fwd5.nunique()>1:out.append((pd.Timestamp(dt),g.factor_value.corr(g.fwd5,method='spearman'),len(g)))
z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date').sort_index(); z.to_csv('scripts/miner_1_20290503_skip5_rawmomentum20_signal.csv')
print('dates',len(z),'avg_n',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4),'recent252',round(z.tail(252).ic.mean(),6),round(z.tail(252).ic.mean()/z.tail(252).ic.std(),6))
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028-29','2028','2029-12-31')]:
 q=z.loc[lo:hi];print(label,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(),6))
print('coverage',round(a.dropna(subset=['factor_value','fwd5']).groupby('date').size().mean()/15,4),'cutoff',a.date.max().date())
