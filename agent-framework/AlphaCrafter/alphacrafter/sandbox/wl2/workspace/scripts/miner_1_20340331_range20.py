import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>250:return x
  except Exception: pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff()
lo=p.rolling(20).min(); hi=p.rolling(20).max(); pos=(p-lo)/(hi-lo+1e-12)
# Short-range overshoot reversal, volatility-normalized and lagged.
vol=r.rolling(20).std(); f=(-pos.rank(axis=1,pct=True))/vol.rank(axis=1,pct=True); f=f.shift(1)
for h in [10,20,40]:
 rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],(p.shift(-h)/p-1).loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); q=z.loc['2026-07-16':'2034-03-30']
 print('h',h,'dates',len(q),'avgN',q.n.mean(),'IC',q.ic.mean(),'ICIR',q.ic.mean()/q.ic.std(ddof=1),'hit',(q.ic>0).mean())
 for lab,qq in [('early',q.loc['2026-07-16':'2028-12-31']),('mid',q.loc['2029':'2031-12-31']),('recent',q.loc['2032':'2034-03-30'])]: print(lab,round(qq.ic.mean(),5),round(qq.ic.mean()/qq.ic.std(ddof=1),4))
print('coverage',f.loc[q.index].notna().mean().mean(),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
