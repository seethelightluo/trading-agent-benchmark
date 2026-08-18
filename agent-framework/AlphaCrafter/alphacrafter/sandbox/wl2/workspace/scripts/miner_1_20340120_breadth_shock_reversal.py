import numpy as np,pandas as pd
from alphacrafter.sim.utils import get_stock_daily_data,get_index_daily_data
U=['000300.SH','SPX','HSI','N225','SX5E','000688.SH','SOX','NDX','XAU','COPPER','WTI','BTC','ETH','US10Y','CN10Y']
def get(s):
 for fn in (get_stock_daily_data,get_index_daily_data):
  try:
   x=fn(s,days=6000)
   if x is not None and len(x)>300:return x
  except Exception: pass
raw={s:get(s) for s in U}; raw={s:x for s,x in raw.items() if x is not None}
p=pd.concat({s:x.set_index('date')['close'] for s,x in raw.items()},axis=1).sort_index(); p.index=pd.to_datetime(p.index); r=np.log(p).diff()
r3=r.rolling(3,min_periods=3).sum(); breadth=(r>0).sum(axis=1)/r.notna().sum(axis=1)
base=-(r3-r3.median(axis=1).to_numpy()[:,None])
mask=(breadth.to_numpy()<0.45)[:,None]
f=pd.DataFrame(base*mask,index=r.index,columns=r.columns).shift(1)
fr=p.shift(-10)/p-1; rows=[]
for d in f.index:
 a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
 if len(a)>=8: rows.append((d,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','ic','n']).set_index('date'); z.index=pd.to_datetime(z.index)
for label,q in [('full',z),('2026_2029',z.loc['2026':'2029']),('2030_2033',z.loc['2030':'2033-12-31']),('recent',z.loc['2029':'2033-12-31'])]:
 print(label,'dates',len(q),'avgN',round(q.n.mean(),3),'IC',round(q.ic.mean(),6),'ICIR',round(q.ic.mean()/q.ic.std(ddof=1),6),'hit',round((q.ic>0).mean(),4))
print('assets',len(raw),'coverage',round(f.notna().mean().mean(),4),'turnover',round(f.rank(axis=1,pct=True).diff().abs().mean(axis=1).mean(),4))
f.to_csv('scripts/miner_1_20340120_breadth_shock_reversal_signal.csv'); z.to_csv('scripts/miner_1_20340120_breadth_shock_reversal_ic.csv')
