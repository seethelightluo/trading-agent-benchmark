import pandas as pd,numpy as np
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_index_daily_data,get_stock_daily_data):
  try:
   d=fn(s,days=5000)
   if d is not None and len(d): return d
  except: pass
px=pd.DataFrame({s:get(s).set_index('date')['close'] for s in U}).sort_index(); r=px.pct_change(); y=px.shift(-1)/px-1
for n in [3,5,10]:
 f=-r.rolling(n).sum()/r.rolling(20).std().replace(0,np.nan)
 rows=[]
 for dt in f.index:
  z=pd.DataFrame({'a':f.loc[dt],'b':y.loc[dt]}).dropna()
  if len(z)>=8 and z.a.nunique()>1: rows.append((dt,z.a.corr(z.b,method='spearman'),len(z)))
 d=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
 print('N',n)
 for label,x in [('all',d),('2020-22',d.loc['2020':'2022']),('2023-24',d.loc['2023':'2024']),('2025-26',d.loc['2025':'2026']),('2027',d.loc['2027':])]:
  q=x.ic.mean();print(label,len(x),round(x.n.mean(),2),round(q,6),round(q/x.ic.std(ddof=1),6),round((x.ic>0).mean(),4))
 print('coverage',f.notna().sum().sum()/(len(U)*len(f)))
