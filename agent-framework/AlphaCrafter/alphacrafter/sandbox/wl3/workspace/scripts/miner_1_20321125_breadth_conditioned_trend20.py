import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
D={}
for s in U:
 x=None
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:x=fn(s,days=5000)
  except Exception:x=None
  if x is not None and len(x):break
 if x is not None and len(x):
  x=x.copy();x.date=pd.to_datetime(x.date);D[s]=x.sort_values('date').drop_duplicates('date').set_index('date').close.astype(float)
p=pd.DataFrame(D).sort_index().ffill(); r=p.pct_change()
# Breadth-conditioned medium trend: 20d relative return, scaled by 30d volatility;
# emphasize trend when market breadth is strong, invert when breadth is weak.
ret20=p/p.shift(20)-1; rel=ret20-ret20.median(axis=1).values[:,None]
vol=r.rolling(30,min_periods=20).std(); breadth=(r.rolling(20).mean()>0).mean(axis=1)
# continuous regime multiplier: positive in broad uptrends, negative in broad downtrends
reg=((breadth-.5)*2).clip(-1,1)
f=(rel/vol*reg.values[:,None]).shift(1)
rows=[]
for i in range(len(p)-11):
 z=pd.concat([f.iloc[i].rename('f'),(p.iloc[i+11]/p.iloc[i+1]-1).rename('y')],axis=1).replace([np.inf,-np.inf],np.nan).dropna()
 if len(z)>=8 and z.f.nunique()>1: rows.append((p.index[i],z.f.corr(z.y),len(z)))
a=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
print('shape',p.shape,'valid_dates',len(a),'assets',len(p.columns),'avg_n',a.n.mean(),'coverage',a.n.sum()/(len(a)*15),'last',p.index[-1].date())
print('IC %.8f ICIR %.8f hit %.4f'%(a.ic.mean(),a.ic.mean()/a.ic.std(ddof=1),(a.ic>0).mean()))
for lo,hi in [('2020','2022'),('2023','2025'),('2026','2028'),('2029','2030'),('2031','2032')]:
 q=a.loc[lo:hi]
 if len(q): print(lo+'-'+hi,'dates',len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
for n in [120,252,756]:
 q=a.tail(n);print('recent',n,'dates',len(q),'IC %.8f ICIR %.8f'%(q.ic.mean(),q.ic.mean()/q.ic.std(ddof=1)))
print('turnover',((f.rank(axis=1,pct=True).diff().abs().mean(axis=1)/2).mean()))
f.reset_index().to_csv('scripts/miner_1_20321125_breadth_conditioned_trend20_signal.csv',index=False)
