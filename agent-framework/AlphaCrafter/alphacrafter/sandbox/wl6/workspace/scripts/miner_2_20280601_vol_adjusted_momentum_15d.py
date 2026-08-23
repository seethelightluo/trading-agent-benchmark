import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def f(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=4000)
   if d is not None and len(d): return d[['date','close']].drop_duplicates('date').set_index('date')['close'].rename(s)
  except FileNotFoundError: pass
 raise RuntimeError(s)
p=pd.concat([f(s) for s in U],axis=1).sort_index().ffill(); r=p.pct_change()
# fully lagged 15-session return divided by 20-session realized volatility
fac=(p.shift(1)/p.shift(16)-1)/(r.shift(1).rolling(20).std()*np.sqrt(20)+1e-12); fw=p.shift(-1)/p-1
rows=[]
for d in fac.index:
 z=pd.concat([fac.loc[d],fw.loc[d]],axis=1).dropna()
 if len(z)>=8: rows.append((d,z.iloc[:,0].corr(z.iloc[:,1],method='spearman'),len(z)))
o=pd.DataFrame(rows,columns=['d','ic','n']).set_index('d'); print('dates',len(o),'avg_n',o.n.mean(),'min_n',o.n.min());print('1d IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for h in [5,10]:
 fw2=p.shift(-h)/p-1; a=[]
 for d in fac.index:
  z=pd.concat([fac.loc[d],fw2.loc[d]],axis=1).dropna()
  if len(z)>=8:a.append(z.iloc[:,0].corr(z.iloc[:,1],method='spearman'))
 a=pd.Series(a);print('%dd IC %.6f ICIR %.6f'%(h,a.mean(),a.mean()/a.std()))
print('coverage',fac.notna().mean().mean(),'turnover',fac.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean())
for lab,a,b in [('2020-22','2020','2022'),('2023-25','2023','2025'),('2026-27','2026','2027'),('2028','2028','2028')]: print(lab,len(o.loc[a:b]),o.loc[a:b,'ic'].mean())
