import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
rows=[]
for s in U:
 d=get_stock_daily_data(s,4000)
 if d is None or len(d)==0:d=get_index_daily_data(s,4000)
 if d is None or len(d)==0: continue
 d=d.sort_values('date'); c=d.close.astype(float); r=c.pct_change()
 x=pd.DataFrame({'date':pd.to_datetime(d.date),'symbol':s,'r3':c.pct_change(3),'vol20':r.rolling(20).std(),'fwd1':c.shift(-1)/c-1,'fwd3':c.shift(-3)/c-1})
 rows.append(x)
a=pd.concat(rows,ignore_index=True)
# Lag vol and cross-sectional demeaning prevent contemporaneous cross-asset level effects
# factor is lagged-volatility-normalized reversal, demeaned each date
base=-a.r3/a.groupby('symbol').vol20.shift(1)
a['factor_value']=base-a.groupby('date')[base.name if base.name else 'r3'].transform('mean') if False else base
# explicit daily demean
cs=a.groupby('date')['factor_value'].transform('mean'); a['factor_value']=a.factor_value-cs
ics={1:[],3:[]}
for dt,g in a.groupby('date'):
 for h,col in [(1,'fwd1'),(3,'fwd3')]:
  g=g.dropna(subset=['factor_value',col])
  if len(g)>=8 and g.factor_value.nunique()>1 and g[col].nunique()>1:
   ics[h].append((pd.Timestamp(dt),g.factor_value.corr(g[col],method='spearman'),len(g)))
for h in ics:
 z=pd.DataFrame(ics[h],columns=['date','ic','n']).set_index('date').sort_index(); z.to_csv('scripts/miner_1_20290322_lagged_volnorm_reversal3_signal.csv' if h==1 else 'scripts/miner_1_20290322_lagged_volnorm_reversal3_h3.csv')
 print('horizon',h,'dates',len(z),'avg_n',round(z.n.mean(),2),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(),6),'hit',round((z.ic>0).mean(),4),'recent252',round(z.tail(252).ic.mean(),6),round(z.tail(252).ic.mean()/z.tail(252).ic.std(),6))
for label,lo,hi in [('2020-22','2020','2022-12-31'),('2023-25','2023','2025-12-31'),('2026-27','2026','2027-12-31'),('2028','2028','2028-12-31')]:
 z=pd.DataFrame(ics[1],columns=['date','ic','n']).set_index('date')
 q=z.loc[lo:hi]; print(label,len(q),round(q.ic.mean(),6),round(q.ic.mean()/q.ic.std(),6))
a.dropna(subset=['factor_value','fwd1']).to_csv('scripts/miner_1_20290322_lagged_volnorm_reversal3_values.csv',index=False)
print('coverage',round(a.dropna(subset=['factor_value','fwd1']).groupby('date').size().mean()/15,4),'cutoff',a.date.max().date())
