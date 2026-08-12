import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def load(s):
 d=None
 try:d=get_index_daily_data(s,days=5000)
 except Exception:pass
 if d is None or len(d)<100:
  try:d=get_stock_daily_data(s,days=5000)
  except Exception:pass
 if d is None:return None
 return d.assign(date=pd.to_datetime(d.date)).drop_duplicates('date').set_index('date').close.astype(float)
p={s:load(s) for s in U};p={s:x for s,x in p.items() if x is not None};px=pd.DataFrame(p).sort_index();r=px.pct_change()
# Stable low-volatility: inverse of blended short/medium realized volatility, lagged through end-of-day.
f=-(.5*r.rolling(10,min_periods=8).std()+.5*r.rolling(30,min_periods=20).std())
def ic(h):
 y=px.pct_change(h).shift(-h);out=[]
 for dt in f.index:
  q=pd.concat([f.loc[dt],y.loc[dt]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:out.append((dt,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
o=ic(1);print('assets',len(p),'dates',len(px),'IC_dates',len(o),'avg_n',round(o.n.mean(),3),'coverage',round(o.n.mean()/15,4));print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a+'-'+b,len(q),q.mean(),q.mean()/q.std() if len(q)>1 else np.nan)
for h in [3,5,10]:
 q=ic(h);print('decay',h,q.ic.mean(),q.ic.mean()/q.ic.std(),len(q))
print('recent120',o.tail(120).ic.mean(),o.tail(120).ic.mean()/o.tail(120).ic.std());f.to_csv('scripts/miner_2_20320708_lowvol_signal.csv',index_label='date')
