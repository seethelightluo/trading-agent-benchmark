import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={s:get_stock_daily_data(s,days=5000) for s in U}; p=pd.concat({s:d.set_index('date').close for s,d in D.items() if d is not None},axis=1).sort_index().ffill(); r=p.pct_change(); out=[]
for t in range(45,len(p)-10):
 f=(p.iloc[t]/p.iloc[t-20]-1)/(r.iloc[t-20:t].where(r.iloc[t-20:t]<0).std().replace(0,np.nan)*np.sqrt(20))
 # penalize recent drawdown and reward positive-day fraction
 f=f*(r.iloc[t-20:t].gt(0).mean())
 z=pd.concat([f.rename('f'),(p.iloc[t+10]/p.iloc[t]-1).rename('r')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8: out.append((p.index[t],z.f.corr(z.r),len(z)))
o=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
for a,b in [('2020','2023'),('2024','2026'),('2027','2029'),('2030','2032')]:
 q=o.loc[a:b]; print(a+'-'+b,len(q),q.ic.mean(),q.ic.mean()/q.ic.std(),(q.ic>0).mean())
print('ALL',len(o),o.n.mean(),o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean())
