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
v20=r.rolling(20).std(); v60=r.rolling(60).std()
f=((-v20/v60).rank(axis=1,pct=True)-r.rolling(60).sum().rank(axis=1,pct=True))/2
f=f.shift(1)
fr=p.shift(-20)/p-1
rows=[]
for d in f.index:
 a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date')
for label,q in [('full',z),('recent_2029_2033',z.loc['2029':'2033-12-23']),('2022_2025',z.loc['2022':'2025']),('2026_2028',z.loc['2026':'2028']),('2030_2033',z.loc['2030':'2033-12-23'])]:
 print(label,'dates',len(q),'avgN',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
print('assets',len(raw),'coverage',round(f.loc['2029':'2033-12-23'].notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).loc['2029':'2033-12-23'].diff().abs().mean(axis=1).mean(),4))
z.to_csv('scripts/miner_3_20331223_compression_reversal_20d_revalidation_ic.csv')
# report annual stability
print('annual')
print(z.groupby(z.index.year).ic.agg(['count','mean']).tail(8).to_string())
