import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>200:return x
  except Exception: pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); r=np.log(p).diff()
r20=r.rolling(20).sum(); med20=r.median(axis=1).rolling(20).sum(); resid=r20.sub(med20,axis=0)
vol=r.rolling(20).std()*np.sqrt(20); breadth=(r20>0).mean(axis=1); gate=((breadth>0.60)|(breadth<0.40)).astype(float)
f=resid.div(vol.replace(0,np.nan)).mul(gate,axis=0).shift(1)
rows=[]
for h in [1,5,10]:
 fr=p.shift(-h)/p-1; rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z.index=pd.to_datetime(z.index); q=z.loc['2026-07-16':'2034-03-30']
 ic=q.ic.mean(); print('H',h,'dates',len(q),'avgN',q.n.mean(),'IC',ic,'ICIR',ic/q.ic.std(ddof=1),'hit',(q.ic>0).mean(),'coverage',f.loc[q.index].notna().mean().mean())
 for lab,qq in [('early',q.loc['2026-07-16':'2028-12-31']),('mid',q.loc['2029':'2031-12-31']),('recent',q.loc['2032':'2034-03-30'])]:
  x=qq.ic.mean(); print(lab,len(qq),x,x/qq.ic.std(ddof=1) if len(qq)>1 else np.nan)
 if h==1:z.to_csv('scripts/miner_2_20340331_breadth_residual_trend_ic.csv')
print('assets',len(raw),'turnover',f.rank(axis=1,pct=True).diff().abs().mean(axis=1).dropna().mean())
f.to_csv('scripts/miner_2_20340331_breadth_residual_trend_signal.csv')
