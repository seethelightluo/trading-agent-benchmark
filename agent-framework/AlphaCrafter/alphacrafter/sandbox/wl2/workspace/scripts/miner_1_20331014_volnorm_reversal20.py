import numpy as np, pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>100:return x
  except Exception: pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index()
r=np.log(p).diff(); v=r.rolling(20).std()
# Higher value = stronger medium-horizon reversal, normalized by current risk.
f=(-p.pct_change(20)/v).rank(axis=1,pct=True).shift(1)
print('assets',len(raw),'dates',len(p),'factor lagged rank(-ret20/vol20)')
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; out=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: out.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(out,columns=['date','ic','n']).set_index('date')
 print('H',h,'dates',len(z),'avgN',round(z.n.mean(),3),'IC',round(z.ic.mean(),6),'ICIR',round(z.ic.mean()/z.ic.std(ddof=1),6),'hit',round((z.ic>0).mean(),4),'coverage',round(len(z)/len(f),4))
 for lo,hi in [('2020','2025-12-31'),('2026','2029-12-31'),('2030','2033-12-31')]:
  zz=z.loc[lo:hi]
  if len(zz): print('REG',lo,len(zz),round(zz.ic.mean(),6),round(zz.ic.mean()/zz.ic.std(ddof=1),6))
 if h==20:z.to_csv('scripts/miner_1_20331014_volnorm_reversal20_20d_ic.csv')
print('turnover',round(f.diff().abs().mean(axis=1).mean(),6),'signal_coverage',round(f.notna().mean().mean(),6))
