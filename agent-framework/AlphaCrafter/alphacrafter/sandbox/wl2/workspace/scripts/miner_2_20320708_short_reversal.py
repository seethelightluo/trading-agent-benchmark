import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=None
 try:d=get_index_daily_data(s,days=5000)
 except:pass
 if d is None or len(d)<100:
  try:d=get_stock_daily_data(s,days=5000)
  except:pass
 if d is None:return None
 return d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close.astype(float)
p={s:load(s) for s in U};p={s:v for s,v in p.items() if v is not None};px=pd.DataFrame(p).sort_index();r=px.pct_change(); vol=r.rolling(20,min_periods=15).std(); f=-px.pct_change(3)/(vol*np.sqrt(3))
def calc(h):
 y=px.pct_change(h).shift(-h);z=[]
 for d in f.index:
  q=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:z.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
 return pd.DataFrame(z,columns=['date','ic','n']).set_index('date')
o=calc(1);print('assets',len(p),'dates',len(px),'IC_dates',len(o),'avg_n',o.n.mean(),'coverage',o.n.mean()/15);print('IC',o.ic.mean(),'ICIR',o.ic.mean()/o.ic.std(),'hit',(o.ic>0).mean())
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a,b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
for h in [3,5,10]:
 q=calc(h);print('decay',h,q.ic.mean(),q.ic.mean()/q.ic.std(),len(q))
print('recent120',o.tail(120).ic.mean(),o.tail(120).ic.mean()/o.tail(120).ic.std());f.to_csv('scripts/miner_2_20320708_short_reversal_signal.csv',index_label='date')
