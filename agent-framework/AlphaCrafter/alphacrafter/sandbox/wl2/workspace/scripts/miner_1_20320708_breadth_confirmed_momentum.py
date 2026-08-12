import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 try:d=get_index_daily_data(s,days=5000)
 except Exception:d=None
 if d is None or len(d)<80:
  try:d=get_stock_daily_data(s,days=5000)
  except Exception:d=None
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d.drop_duplicates('date').set_index('date').close.astype(float)
p={s:get(s) for s in U};p={s:x for s,x in p.items() if x is not None}; px=pd.DataFrame(p).sort_index(); r=px.pct_change()
# Breadth-confirmed risk-adjusted medium momentum: 10d return / 20d vol, activated smoothly
vol=r.rolling(20).std(); raw=px.pct_change(10)/(vol*np.sqrt(10)); breadth=(r.rolling(5).mean()>0).mean(axis=1)
# smooth confirmation, using only information at date t
f=raw.mul((1/(1+np.exp(-12*(breadth-0.5)))),axis=0)
def ic(h):
 fr=px.pct_change(h).shift(-h); out=[]
 for d in f.index:
  q=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1: out.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
 return pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
o=ic(1); print('assets',len(p),'dates',len(px),'IC_dates',len(o),'avg_n',round(o.n.mean(),2),'coverage',round(o.n.mean()/15,4)); print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2020','2022'),('2023','2025'),('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic; print(a+'-'+b,'dates',len(q),'IC %.6f ICIR %.6f hit %.4f'%(q.mean(),q.mean()/q.std() if len(q)>1 else np.nan,(q>0).mean() if len(q) else np.nan))
for h in [3,5,10]:
 q=ic(h); print('decay',h,'IC %.6f IC_dates %d'%(q.ic.mean(),len(q)))
q=o.tail(120).ic;print('recent120','IC %.6f ICIR %.6f dates %d'%(q.mean(),q.mean()/q.std(),len(q)))
f.to_csv('scripts/miner_1_20320708_breadth_confirmed_momentum_signal.csv',index_label='date')
