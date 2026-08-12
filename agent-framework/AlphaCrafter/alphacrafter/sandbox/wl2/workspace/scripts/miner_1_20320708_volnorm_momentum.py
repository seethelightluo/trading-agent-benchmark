import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_index_daily_data,get_stock_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def g(s):
 try:d=get_index_daily_data(s,5000)
 except Exception:d=None
 if d is None or len(d)<80:
  try:d=get_stock_daily_data(s,5000)
  except Exception:d=None
 if d is None:return None
 d=d.copy();d.date=pd.to_datetime(d.date);return d.drop_duplicates('date').set_index('date').close.astype(float)
p={s:g(s) for s in U};p={s:x for s,x in p.items() if x is not None};px=pd.DataFrame(p).sort_index();r=px.pct_change();v=r.rolling(20).std()
# Intermediate horizon risk-adjusted momentum, with a volatility-regime dampener
raw=px.pct_change(10)/(v*np.sqrt(10)); vr=(v/v.rolling(60).median()).clip(.25,4); f=raw/(1+0.5*(vr-1).clip(lower=0))
def test(h):
 y=px.pct_change(h).shift(-h);a=[]
 for d in f.index:
  q=pd.concat([f.loc[d],y.loc[d]],axis=1).dropna()
  if len(q)>=8 and q.iloc[:,0].nunique()>1 and q.iloc[:,1].nunique()>1:a.append((d,q.iloc[:,0].rank().corr(q.iloc[:,1].rank()),len(q)))
 return pd.DataFrame(a,columns=['date','ic','n']).set_index('date')
o=test(1);print('assets',len(p),'dates',len(px),'IC_dates',len(o),'avg_n',round(o.n.mean(),2),'coverage',round(o.n.mean()/15,4));print('IC %.6f ICIR %.6f hit %.4f'%(o.ic.mean(),o.ic.mean()/o.ic.std(),(o.ic>0).mean()))
for a,b in [('2026','2029'),('2030','2032')]:
 q=o.loc[a:b].ic;print(a+'-'+b,len(q),round(q.mean(),6),round(q.mean()/q.std(),6))
for h in [3,5,10]:q=test(h);print('decay',h,round(q.ic.mean(),6),len(q))
q=o.tail(120).ic;print('recent',round(q.mean(),6),round(q.mean()/q.std(),6));f.to_csv('scripts/miner_1_20320708_volnorm_momentum_signal.csv',index_label='date')
