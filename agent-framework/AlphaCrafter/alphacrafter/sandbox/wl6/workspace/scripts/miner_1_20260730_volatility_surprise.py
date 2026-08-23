import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data,get_account_dict
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
x={}
for s in U:
 try: d=get_index_daily_data(s, days=2500)
 except Exception:
  try: d=get_stock_daily_data(s, days=2500)
  except Exception as e: print('skip',s); continue
 if d is None or len(d)<100: continue
 d=d.copy(); d['r']=pd.to_numeric(d['close']).pct_change()
 d['f']=-(d.r.rolling(5,min_periods=5).std()/d.r.rolling(60,min_periods=45).std()).replace([np.inf,-np.inf],np.nan)
 x[s]=d.set_index(pd.to_datetime(d.date))
rows=[]
for s,d in x.items():
 z=d.loc['2020-01-01':'2026-07-15']
 for j,dt in enumerate(z.index[:-1]):
  if pd.notna(z.f.iloc[j]) and pd.notna(z.r.iloc[j+1]): rows.append((dt,s,z.f.iloc[j],z.r.iloc[j+1]))
a=pd.DataFrame(rows,columns=['date','s','f','y'])
ics=a.groupby('date').apply(lambda q:q.f.corr(q.y,method='spearman') if len(q)>=8 else np.nan).dropna()
print('available',len(x),'dates',len(ics),'avg_n',a.groupby('date').size().mean(),'coverage',len(a)/(len(ics)*15),'IC',ics.mean(),'ICIR',ics.mean()/ics.std(ddof=1),'hit',(ics>0).mean())
for yr,g in ics.groupby(ics.index.year): print(yr, len(g),g.mean(),g.mean()/g.std(ddof=1))
r=a.assign(rank=a.groupby('date').f.rank(pct=True)).pivot(index='date',columns='s',values='rank')
print('turnover',r.diff().abs().mean().mean())
