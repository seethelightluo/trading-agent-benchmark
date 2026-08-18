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
# Candidate: distance from 60-day range midpoint, reversal. A bounded location signal
lo=p.rolling(60,min_periods=45).min(); hi=p.rolling(60,min_periods=45).max()
loc=(2*p-lo-hi)/(hi-lo)
f=(-loc).shift(1)
rows=[]
for d in f.index:
 for h in [5,10,20]:
  fr=p.shift(-h)/p-1; a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,h,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [5,10,20]:
 q=z[z.h==h].set_index('date')
 for label,qq in [('full',q),('2026_2028',q.loc['2026':'2028']),('2029_2033',q.loc['2029':'2033-12-31']),('2031_2033',q.loc['2031':'2033-12-31'])]:
  ic=qq.ic.mean(); ir=ic/qq.ic.std(ddof=1)
  print(h,label,'dates',len(qq),'avgN',round(qq.n.mean(),3),'IC',round(ic,6),'ICIR',round(ir,6),'hit',round((qq.ic>0).mean(),4))
q=z[z.h==10].set_index('date'); rr=f.loc['2029':'2033-12-31']
print('assets',len(raw),'coverage',round(f.loc['2029':'2033-12-31'].notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).loc['2029':'2033-12-31'].diff().abs().mean(axis=1).mean(),4))
z.to_csv('scripts/miner_2_20340120_range_location_reversal_ic.csv',index=False)
