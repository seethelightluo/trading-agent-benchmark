import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']; fs={}
for s in U:
 d=get_stock_daily_data(s,days=2600); fs[s]=d[['date','close']].set_index('date') if d is not None else None
for win in [5,10,20,40,60]:
 rows=[]
 for s,d in fs.items():
  if d is None: continue
  r=np.log(d.close).diff(); f=-(r.rolling(win).std().shift(1)); y=np.log(d.close.shift(-10)/d.close)
  rows.append(pd.DataFrame({'f':f,'y':y}).dropna())
 x=pd.concat(rows); z=[]
 for dt,g in x.groupby(level=0):
  if len(g)>=8:z.append(g.f.corr(g.y))
 a=pd.Series(z).dropna(); print(win,len(a),a.mean(),a.mean()/a.std(ddof=1),(a>0).mean())
