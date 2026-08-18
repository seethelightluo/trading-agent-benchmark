import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
px={}
for s in U:
 d=get_stock_daily_data(s,days=3000); z=d[['date','close']].copy(); z.date=pd.to_datetime(z.date); px[s]=z.drop_duplicates('date').set_index('date').close
close=pd.DataFrame(px).sort_index(); ret=close.pct_change()
# Breadth/momentum persistence: fraction of positive daily observations over trailing 20 days,
# centered by cross-sectional median to isolate relative persistence.
breadth=ret.gt(0).rolling(20,min_periods=15).mean()
fac=breadth.sub(breadth.median(axis=1),axis=0)
for h in [1,5,10]:
 fw=close.pct_change(h).shift(-h); vals=[]; ns=[]; dates=[]
 for dt in fac.index:
  a=pd.DataFrame({'f':fac.loc[dt],'r':fw.loc[dt]}).dropna()
  if len(a)>=8 and a.f.nunique()>1 and a.r.nunique()>1:
   vals.append(a.f.corr(a.r)); ns.append(len(a)); dates.append(dt)
 ic=pd.Series(vals,index=dates).dropna(); print('h',h,'dates',len(ic),'meanN',np.mean(ns),'IC %.6f ICIR %.6f hit %.4f'%(ic.mean(),ic.mean()/ic.std(ddof=1)*np.sqrt(252),(ic>0).mean()))
 for yr in ['2020','2021','2022','2023','2024','2025','2026']:
  q=ic[[str(x)[:4]==yr for x in ic.index]]; print(yr, '%.5f'%q.mean() if len(q) else 'nan',len(q))
r=fac.rank(axis=1,pct=True); print('turnover',r.diff().abs().mean(axis=1).mean(),'coverage',fac.notna().sum().sum()/fac.size,'period',close.index.min(),close.index.max())
print('decay done; 15 instruments, >=8/date')
