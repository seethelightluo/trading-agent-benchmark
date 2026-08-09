import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_account_dict,get_stock_daily_data
U=get_account_dict().get('watch_list',[]) or ['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
m=pd.read_csv('../persistent/index_data/DXY.csv',parse_dates=['date']).set_index('date').close.pct_change().loc[:'2026-07-15']
rets={}
for s in U:
 d=get_stock_daily_data(s,days=3000)
 if d is not None:
  x=d.copy(); x.date=pd.to_datetime(x.date); rets[s]=x.set_index('date').close.pct_change().loc[:'2026-07-15']
r=pd.DataFrame(rets).sort_index(); ix=r.index.intersection(m.index); r=r.loc[ix]; m=m.loc[ix]
bet=r.rolling(60,min_periods=45).cov(m).div(m.rolling(60,min_periods=45).var(),axis=0); f=-bet
rows=[]; dates=[]
for i in range(len(r)-1):
 z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1].rename('y')],axis=1).dropna()
 if len(z)>=8: rows.append(z.f.corr(z.y)); dates.append(r.index[i])
ic=pd.Series(rows,index=pd.to_datetime(dates)); print('universe',len(U),'dates',len(ic),'mean_names',len(U),'coverage',f.notna().sum().sum()/f.size); print('daily IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(),(ic>0).mean()))
for h in [5,10]:
 a=[]
 for i in range(len(r)-h):
  z=pd.concat([f.iloc[i].rename('f'),r.iloc[i+1:i+1+h].sum().rename('y')],axis=1).dropna()
  if len(z)>=8:a.append(z.f.corr(z.y))
 a=pd.Series(a).dropna(); print('%dd IC %.6f ICIR %.6f n %d'%(h,a.mean(),a.mean()/a.std(),len(a)))
for a,b in [('2020','2022'),('2023','2024'),('2025','2026')]:
 x=ic.loc[(ic.index>=a)&(ic.index<=b)]; print(a,b,'IC %.5f ICIR %.5f n %d'%(x.mean(),x.mean()/x.std(),len(x)))
print('turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for name,other in [('reversal',-r.shift(1).rolling(5).sum()),('momentum',r.rolling(20).mean()),('clv',r.rolling(3).mean())]:
 z=pd.concat([f.stack().rename('a'),other.reindex_like(f).stack().rename('b')],axis=1).dropna(); print('corr',name,z.a.corr(z.b))
