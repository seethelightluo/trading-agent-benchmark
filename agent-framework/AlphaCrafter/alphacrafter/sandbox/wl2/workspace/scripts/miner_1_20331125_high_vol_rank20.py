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
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index()
r=np.log(p).diff(); vol=r.rolling(20).std(); f=vol.rank(axis=1,pct=True).shift(1)
f.to_csv('scripts/miner_1_20331125_high_vol_rank20_signal.csv')
print('candidate high_vol_rank20 assets',len(raw),'dates',len(p),'signal_rows',f.notna().any(axis=1).sum())
for h in [5,10,20]:
 fr=p.shift(-h)/p-1; rows=[]
 for d in f.index:
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
 z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); ic=z.ic.mean(); ir=ic/z.ic.std(ddof=1)
 print('H',h,'dates',len(z),'avgN',round(z.n.mean(),3),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((z.ic>0).mean(),4))
print('coverage',round(f.notna().mean().mean(),6),'turnover',round(f.diff().abs().mean(axis=1).mean(),6))
