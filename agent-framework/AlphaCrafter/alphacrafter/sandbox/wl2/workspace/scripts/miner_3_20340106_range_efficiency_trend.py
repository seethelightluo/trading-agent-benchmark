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
r=np.log(p).diff()
# Range-efficiency trend: directional displacement divided by path length;
# rewards persistent trends and avoids raw magnitude duplication.
eff=r.rolling(20).sum()/r.abs().rolling(20).sum()
f=eff.shift(1)
rows=[]
for d in f.index:
 for h in [10,20]:
  fr=p.shift(-h)/p-1
  a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,h,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','h','ic','n'])
for h in [10,20]:
 q=z[z.h==h].set_index('date')
 for label,qq in [('full',q),('2029_2033',q.loc['2029':'2033-12-31']),('2030_2033',q.loc['2030':'2033-12-31']),('2026_2028',q.loc['2026':'2028'])]:
  print(h,label,'dates',len(qq),'avgN',round(qq.n.mean(),3),'IC',round(qq.ic.mean(),6),'ICIR',round(qq.ic.mean()/qq.ic.std(ddof=1),6),'hit',round((qq.ic>0).mean(),4))
q=z[z.h==20].set_index('date')
print('assets',len(raw),'coverage',round(f.loc['2029':'2033-12-31'].notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).loc['2029':'2033-12-31'].diff().abs().mean(axis=1).mean(),4))
print('annual20'); print(q.groupby(q.index.year).ic.agg(['count','mean']).tail(8).to_string())
z.to_csv('scripts/miner_3_20340106_range_efficiency_trend_ic.csv',index=False)
