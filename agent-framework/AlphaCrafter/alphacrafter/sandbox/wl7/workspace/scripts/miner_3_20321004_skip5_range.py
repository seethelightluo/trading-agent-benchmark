import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; cut=pd.Timestamp('2032-10-03'); D={}
for s in U:
 x=get_stock_daily_data(s,days=5000)
 if x is not None and len(x):
  x=x.copy(); x.date=pd.to_datetime(x.date); x=x[x.date<=cut].sort_values('date').drop_duplicates('date'); D[s]=x.set_index('date')
rows=[]
for s,x in D.items():
 c=x.close.astype(float); tr=(x.high-x.low)/c.replace(0,np.nan); scale=tr.rolling(30,min_periods=20).median(); r5=c.pct_change(5).shift(5); r60=c.pct_change(60).shift(5)
 f=(-r5/scale.shift(5))*(1+.35*np.tanh(-r60/.15))
 for h in [1,5,10,20]: rows.append(pd.DataFrame({'date':c.index,'factor':f,'fwd':c.shift(-h)/c-1,'h':h}).reset_index(drop=True))
a=pd.concat(rows,ignore_index=True).replace([np.inf,-np.inf],np.nan).dropna()
for h,g0 in a.groupby('h'):
 z=[]
 for d,g in g0.groupby('date'):
  if len(g)>=8 and g.factor.nunique()>2 and g.fwd.nunique()>2:z.append(g.factor.corr(g.fwd))
 v=pd.Series(z).dropna(); print('H',h,'dates',len(v),'IC',v.mean(),'ICIR',v.mean()/v.std(ddof=1),'hit',(v>0).mean(),'thirds',[v.iloc[i*len(v)//3:(i+1)*len(v)//3].mean() for i in range(3)])
 if h==10: g0[['date','factor']].to_csv('scripts/miner_3_20321004_skip5_range_signal.csv',index=False)
print('instruments',len(D))
