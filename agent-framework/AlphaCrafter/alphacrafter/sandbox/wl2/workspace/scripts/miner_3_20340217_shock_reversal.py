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
# Volatility-normalized short-term reversal, with a slow trend agreement tilt.
# Reversal is strongest after unusually large recent moves; slow trend term avoids
# treating persistent directional trends as pure mean-reversion. Entire signal lagged.
short=r.rolling(5).sum()
vol=r.rolling(20).std()*np.sqrt(20)
shock=(short/(vol+1e-12)).clip(-4,4)
trend=r.rolling(60).sum()/(r.abs().rolling(60).sum()+1e-12)
f=(-shock.rank(axis=1,pct=True))*((1-trend.abs()).clip(0,1).rank(axis=1,pct=True))
f=f.shift(1)
rows=[]
for d in f.index:
 for h in [10,20,40]:
  fr=p.shift(-h)/p-1; a=pd.concat([f.loc[d],fr.loc[d]],axis=1).dropna()
  if len(a)>=8: rows.append((d,h,a.iloc[:,0].corr(a.iloc[:,1]),len(a)))
z=pd.DataFrame(rows,columns=['date','h','ic','n']); z.date=pd.to_datetime(z.date)
for h in [10,20,40]:
 q=z[z.h==h].set_index('date')
 for label,qq in [('full',q),('2026_2028',q.loc['2026':'2028']),('2029_2033',q.loc['2029':'2033-12-31']),('2031_2034',q.loc['2031':'2034-02-16'])]:
  if len(qq)>1: print(h,label,'dates',len(qq),'avgN',round(qq.n.mean(),3),'IC',round(qq.ic.mean(),6),'ICIR',round(qq.ic.mean()/qq.ic.std(ddof=1),6),'hit',round((qq.ic>0).mean(),4))
q=z[z.h==20].set_index('date')
print('assets',len(raw),'coverage',round(f.loc['2026':'2034-02-16'].notna().mean().mean(),4),'active_date_fraction',round(f.notna().any(axis=1).mean(),4))
print('annual20'); print(q.groupby(q.index.year).ic.agg(['count','mean']).tail(8).to_string())
z.to_csv('scripts/miner_3_20340217_shock_reversal_ic.csv',index=False)
f.to_csv('scripts/miner_3_20340217_shock_reversal_signal.csv')
